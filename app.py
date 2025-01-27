from flask import Flask, render_template, Response
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import os
import webbrowser
import time
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Load YOLO model
model = YOLO('best-fire.pt')  # Path to your YOLOv8 model

rtsp = "rtsp://192.168.0.2:554/PSIA/Streaming/channels/0"
cap = cv2.VideoCapture(rtsp)
# Initialize webcam
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible!")

lock = threading.Lock()

# Create a directory to store detections
detection_dir = "detections"
os.makedirs(detection_dir, exist_ok=True)

def generate_frames():
    """Generate video frames for streaming."""
    last_processed_time = 0
    while True:
        with lock:
            ret, frame = cap.read()
            if not ret:
                continue

            # Limit processing to one frame per second
            current_time = time.time()
            if current_time - last_processed_time >= 1:
                last_processed_time = current_time

                # Process frame with YOLO
                results = model.predict(source=frame, save=False, conf=0.5)

                for box in results[0].boxes:
                    class_id = int(box.cls)
                    confidence = box.conf.item()
                    if class_id == 0:  # Assuming class 0 is fire
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                        cv2.putText(
                            frame,
                            f"Fire: {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        # Save detection to file
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
                        filename = os.path.join(detection_dir, f"fire_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)

                        # Emit log message to UI
                        log_message = {
                            'message': f"Fire detected with confidence {confidence:.2f}",
                            'time': timestamp
                        }
                        print(f"Emitting log: {log_message}")  # Debugging log
                        socketio.emit('log', log_message)


            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print("Client connected!")

if __name__ == "__main__":
    # Open the default web browser automatically
    webbrowser.open('http://127.0.0.1:5000')

    # Run the app
    socketio.run(app, host='0.0.0.0', port=5000)
