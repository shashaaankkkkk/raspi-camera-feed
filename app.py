from flask import Flask, render_template, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

# Initialize the PiCamera2 instance
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format":'RGB888',"size": (640, 480)})
picam2.configure(config)
picam2.start()

def generate_frames():
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
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
    app.run(host='0.0.0.0', port=5000, debug=False)
