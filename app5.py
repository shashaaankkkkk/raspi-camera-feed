from flask import Flask, render_template, Response, jsonify
from picamera2 import Picamera2
import cv2
import numpy as np
import time
from threading import Thread, Lock

app = Flask(__name__)

# Initialize the PiCamera2 instance
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (416, 416)})
picam2.configure(config)
picam2.start()

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variables
frame_buffer = None
latest_detections = []
buffer_lock = Lock()
detection_lock = Lock()

# Configuration
DETECTION_INTERVAL = 0.5  # Perform detection every 0.5 seconds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold

def capture_frames():
    global frame_buffer
    while True:
        frame = picam2.capture_array()
        frame = cv2.resize(frame, (416, 416))  # Resize to YOLOv4-tiny input size
        with buffer_lock:
            frame_buffer = frame

def perform_detection():
    global latest_detections, frame_buffer
    last_detection_time = 0
    while True:
        current_time = time.time()
        if current_time - last_detection_time > DETECTION_INTERVAL:
            with buffer_lock:
                if frame_buffer is None:
                    continue
                frame = frame_buffer.copy()
            
            # Prepare image for YOLOv4-tiny
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            
            # Perform YOLOv4-tiny detection
            outs = net.forward(output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > CONFIDENCE_THRESHOLD:
                        center_x = int(detection[0] * 416)
                        center_y = int(detection[1] * 416)
                        w = int(detection[2] * 416)
                        h = int(detection[3] * 416)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            
            detections = []
            for i in indices:
                i = i[0] if isinstance(i, (list, np.ndarray)) else i
                box = boxes[i]
                x, y, w, h = box
                detections.append({
                    'class': classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': [x, y, x+w, y+h]
                })
            
            with detection_lock:
                latest_detections = detections
            
            last_detection_time = current_time

def generate_frames():
    global frame_buffer, latest_detections
    while True:
        with buffer_lock:
            if frame_buffer is None:
                continue
            frame = frame_buffer.copy()
        
        with detection_lock:
            detections = latest_detections.copy()
        
        # Draw bounding boxes on the frame
        for detection in detections:
            label = detection['class']
            conf = detection['confidence']
            box = detection['box']
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
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    with detection_lock:
        detections = latest_detections.copy()
    # Filter detections to include only persons and objects they might be holding
    filtered_detections = [
        det for det in detections
        if det['class'] == 'person' or (det['class'] != 'person' and any(
            is_object_held_by_person(det, person_det)
            for person_det in detections if person_det['class'] == 'person'
        ))
    ]
    return jsonify(filtered_detections)

def is_object_held_by_person(obj_det, person_det):
    obj_box = obj_det['box']
    person_box = person_det['box']
    obj_area = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])
    intersection_area = max(0, min(obj_box[2], person_box[2]) - max(obj_box[0], person_box[0])) * \
                        max(0, min(obj_box[3], person_box[3]) - max(obj_box[1], person_box[1]))
    return intersection_area / obj_area > 0.5

if __name__ == '__main__':
    # Start background threads
    Thread(target=capture_frames, daemon=True).start()
    Thread(target=perform_detection, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
