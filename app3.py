from flask import Flask, render_template, Response
from picamera2 import Picamera2
import cv2
import numpy as np

app = Flask(__name__)

# Initialize the PiCamera2 instance
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Set camera controls for better color grading
picam2.set_controls({
    "AwbMode": "auto",         # Auto White Balance
    "Brightness": 0.5,         # Adjust brightness
    "Contrast": 1.0,           # Adjust contrast
    "Saturation": 1.0,         # Adjust saturation
    "ExposureTime": 50000      # Adjust exposure time (in microseconds)
    # Add more controls as needed
})

# Function to apply gamma correction
def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def generate_frames():
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()

        # Apply gamma correction to the frame
        frame = adjust_gamma(frame, gamma=1.5)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        frame = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Render the HTML page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
