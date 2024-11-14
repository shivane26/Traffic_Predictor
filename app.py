from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2

# Initialize Flask app
app = Flask(__name__)

# Set up upload folder using a relative path
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

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
            return render_template('index.html', video_url=url_for('static', filename=f'uploads/{filename}'))

    return render_template('index.html')

# Function to process the uploaded video and detect traffic signs
def process_video(video_path):
    # Video processing logic remains here...
    pass

# Run the app with dynamic port and host
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
