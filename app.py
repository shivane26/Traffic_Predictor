from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import logging
from ultralytics import YOLO
import cv2

# Initialize Flask app
app = Flask(__name__)

# Set up upload folder using a relative path
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set a max file size (100 MB as an example)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO('traffic_sign_predictor.pt')

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
    try:
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
                return render_template('index.html', video_url=url_for('static', filename=f'uploads/{processed_video_path}'))

    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return "Internal Server Error", 500

    return render_template('index.html')

# Function to process the uploaded video and detect traffic signs
def process_video(video_path):
    try:
        # Load the video
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            logger.error("Error opening video file")
            return None
        
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))

        # Define the codec and create a VideoWriter object for saving the processed video
        processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(video_path))
        video_writer = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Resize frame before passing it to the YOLO model (optional, reduces memory load)
            frame_resized = cv2.resize(frame, (640, 480))  # Resize to lower resolution
            results = model.predict(frame_resized)

            # Draw bounding boxes and labels on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates for the bounding box
                    confidence = box.conf[0]  # Confidence score
                    class_id = int(box.cls[0])  # Class ID

                    # Draw rectangle and add label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the processed frame to the output video
            video_writer.write(frame)

        # Release resources
        video_capture.release()
        video_writer.release()

        logger.info(f"Processed video saved at {processed_video_path}")
        return processed_video_path
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        return None

# Run the app with dynamic port and host
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
