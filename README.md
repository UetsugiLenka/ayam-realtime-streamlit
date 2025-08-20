# 🍗 Real-time Deteksi Bagian & Kesegaran Ayam

Skripsi ini mengimplementasikan **two-stage pipeline**:
1. **Deteksi Bagian Ayam** menggunakan YOLOv8
2. **Klasifikasi Kesegaran Ayam** menggunakan MobileNetV3-Large

Aplikasi ini berjalan **real-time** di web browser menggunakan [Streamlit](https://streamlit.io) + [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc).

---

## 🚀 Demo
Aplikasi ini sudah dideploy di **Streamlit Cloud** dan dapat diakses melalui link berikut:

👉 [Demo App (klik di sini)](https://ayam-realtime-streamlit.streamlit.app)

📱 Bisa diakses dari:
- Laptop (pakai webcam)
- Handphone (pakai kamera HP via browser)

---

## 🧑‍💻 Cara Menjalankan Lokal

### 1. Clone repo
```bash
git clone https://github.com/USERNAME/ayam-realtime-streamlit.git
cd ayam-realtime-streamlit
