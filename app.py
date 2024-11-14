from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import tempfile

# Initialize Flask app
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Load YOLO model
model = YOLO('/Users/shiva/Desktop/FOML_Project/traffic_sign_predictor.pt')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route for video
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        
        file = request.files['video']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded video and detect traffic signs
            processed_video_path = process_video(filepath)

            # Return the processed video URL to display it in the frontend
            return render_template('index.html', video_url=url_for('static', filename='uploads/processed_video.mp4'))

    return render_template('index.html')  # Just show the upload form if GET request



# Function to process the uploaded video and detect traffic signs
def process_video(video_path):
    # Open the uploaded video
    cap = cv2.VideoCapture(video_path)

    # Get video properties like width, height, and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a temporary file to save the processed video
    processed_video_path = 'static/uploads/processed_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 on the frame
        results = model(frame)

        # Draw bounding boxes on the frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    return processed_video_path

if __name__ == '__main__':
    app.run(debug=True)
