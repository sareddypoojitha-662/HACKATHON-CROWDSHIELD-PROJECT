import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO
import webbrowser
import threading
import time

# Initialize Flask application.
app = Flask(__name__)

# Load the pre-trained YOLOv8 model.
model = YOLO('yolov8n.pt')

# Variable to hold the person count, accessible globally.
person_count = 0

# Generator function to capture frames and perform detection.
def generate_frames():
    global person_count
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform object detection on the frame.
        results = model(frame, stream=True)
        
        # Process detection results.
        current_person_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if model.names[int(box.cls[0])] == 'person':
                    current_person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Update the global count.
        person_count = current_person_count
        
        # Add the count to the frame for display.
        cv2.putText(frame, f'People: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode the processed frame as a JPEG image.
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame = buffer.tobytes()
        # Yield the frame as a byte stream for the web server.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data_feed():
    return f'People Count: {person_count}'

# The initial page with the "Go Live" button, now at the new URL.
@app.route('/evofluxcrowdshield')
def start_page():
    return render_template('start_page.html')

# The dashboard with the live video, now at a new URL.
@app.route('/dashboard')
def live_dashboard():
    return render_template('index.html')
# Add this new route below your other routes.
# @app.route('/test-image')
# def test_image():
#     return app.send_static_file('background.jpg')

def open_browser():
    time.sleep(3)
    # The browser will now open the new URL for the button page.
    webbrowser.open_new_tab('http://127.0.0.1:5001/evofluxcrowdshield')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=5001, debug=True)