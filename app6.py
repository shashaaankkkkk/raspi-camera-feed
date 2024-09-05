from flask import Flask, render_template, Response, jsonify
from picamera2 import Picamera2
import cv2
import torch
import numpy as np

app = Flask(__name__)

# Initialize the PiCamera2 instance
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Global variable to store the latest detections
latest_detections = []

def generate_frames():
    global latest_detections
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()
       
        # Perform YOLOv5 detection
        results = model(frame)
       
        # Update latest_detections
        latest_detections = results.pandas().xyxy[0].to_dict('records')
       
        # Draw bounding boxes on the frame
        for detection in latest_detections:
            label = detection['name']
            conf = detection['confidence']
            box = [int(detection['xmin']), int(detection['ymin']),
                   int(detection['xmax']), int(detection['ymax'])]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
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

@app.route('/api/detections')
def get_detections():
    global latest_detections
    # Filter detections to include only persons and objects they might be holding
    filtered_detections = [
        det for det in latest_detections
        if det['name'] == 'person' or (det['name'] != 'person' and any(
            is_object_held_by_person(det, person_det)
            for person_det in latest_detections if person_det['name'] == 'person'
        ))
    ]
    return jsonify(filtered_detections)

def is_object_held_by_person(obj_det, person_det):
    # Simple heuristic: check if the object's bounding box is mostly inside the person's bounding box
    obj_area = (obj_det['xmax'] - obj_det['xmin']) * (obj_det['ymax'] - obj_det['ymin'])
    intersection_area = max(0, min(obj_det['xmax'], person_det['xmax']) - max(obj_det['xmin'], person_det['xmin'])) * \
                        max(0, min(obj_det['ymax'], person_det['ymax']) - max(obj_det['ymin'], person_det['ymin']))
    return intersection_area / obj_area > 0.5

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
