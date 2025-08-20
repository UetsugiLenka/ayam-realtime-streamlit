import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Deteksi Ayam Real-time", layout="wide")

# ========== Load Models ==========
@st.cache_resource
def load_models():
    yolo = YOLO("yolo_chicken_parts.pt")   # model YOLO deteksi bagian ayam
    freshness = load_model("mobilenetv3_freshness.keras")  # MobileNet klasifikasi kesegaran
    return yolo, freshness

yolo_model, freshness_model = load_models()
freshness_classes = ["Busuk", "Busuk-Setengah", "Segar"]
color_map = {"Busuk": (0, 0, 255), "Busuk-Setengah": (0, 255, 255), "Segar": (0, 255, 0)}

# ========== Video Processor ==========
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = yolo_model(img, imgsz=320, verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, box)
                crop_img = img[y1:y2, x1:x2]

                if crop_img.size == 0:
                    continue

                # Preprocess untuk MobileNet
                crop_resized = cv2.resize(crop_img, (224, 224))
                crop_resized = crop_resized.astype("float32") / 255.0
                crop_resized = np.expand_dims(crop_resized, axis=0)

                pred = freshness_model.predict(crop_resized, verbose=0)
                class_id = np.argmax(pred)
                freshness_label = freshness_classes[class_id]
                score = np.max(pred)

                part_label = yolo_model.names[label]
                final_label = f"{part_label} - {freshness_label} ({score:.2f})"

                color = color_map[freshness_label]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, final_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========== UI Streamlit ==========
st.title("🍗 Real-time Deteksi Bagian & Kesegaran Ayam")
st.markdown("YOLO untuk deteksi bagian ayam + MobileNetV3 untuk klasifikasi kesegaran.")

webrtc_streamer(
    key="ayam-stream",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
