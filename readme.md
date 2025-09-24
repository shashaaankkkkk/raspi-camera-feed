## README.md

# Raspberry Pi Camera Stream with Flask & PiCamera2

This project streams live video from the Raspberry Pi Camera Module using the **PiCamera2** library and serves it over a **Flask web server**. The stream is displayed in the browser with near-zero latency using **MJPEG streaming**.

---

##  Features

* Real-time camera feed using PiCamera2.
* Flask backend serving the stream.
* Browser access at `/video_feed`.
* Adjustable resolution and format.
* Lightweight, \~0ms latency on local network.

---

##  Requirements

* Raspberry Pi (with 64-bit OS recommended).
* Raspberry Pi Camera Module enabled (`raspi-config`).
* Python 3.9+
* Dependencies:

  ```bash
  pip install flask opencv-python picamera2
  ```

---

##  Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/pi-camera-stream.git
   cd pi-camera-stream
   ```

2. Run the app:

   ```bash
   python app.py
   ```

3. Open browser and visit:

   ```
   http://<raspberry-pi-ip>:5000
   ```

---

##  Project Structure

```
pi-camera-stream/
│── app.py          # Flask server with PiCamera2 video stream
│── templates/
│     └── index.html  # Basic UI to display the video feed
│── README.md
```

---

##  Endpoints

* `/` → Webpage with video feed
* `/video_feed` → Raw MJPEG stream

---

##  Configuration

* Default resolution: **640x480**
* Format: **RGB888**
* To change resolution, edit in `app.py`:

  ```python
  config = picam2.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
  ```

---

##  Notes

* Use wired LAN or local WiFi for lowest latency.
* Works best in Chrome/Edge/Firefox.
* Add **ngrok / reverse proxy** if you want to view feed remotely.

---

##  Future Enhancements

* Add motion detection (OpenCV).
* Record and save stream to file.
* Add authentication for `/video_feed`.

---

##  Python Script Description

The script does the following:

1. Initializes PiCamera2 and configures resolution/format.
2. Starts the camera and continuously captures frames.
3. Encodes each frame as JPEG.
4. Streams frames over Flask using `multipart/x-mixed-replace`.
5. Provides `/` endpoint for the UI and `/video_feed` endpoint for direct MJPEG stream.

---
