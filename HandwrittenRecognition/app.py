from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import pytesseract

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = r'C:\Users\Hp\Desktop\PROJECTS\HandwrittenRecognition\detected'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model_path = hf_hub_download(local_dir=".",
                             repo_id="armvectores/yolov8n_handwritten_text_detection",
                             filename="best.pt")
model = YOLO(model_path)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        return redirect(url_for('upload_file'))  # Redirect to upload page after login
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup logic here, e.g., save user data to a database
        # After processing, redirect to login page
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            detected_file_path = os.path.join(DETECTED_FOLDER, filename)
            res = model.predict(source=file_path, project='.', name='detected', exist_ok=True, 
                                save=True, show=False, show_labels=False, show_conf=False, conf=0.5)

            if os.path.exists(detected_file_path):
                detected_image = Image.open(detected_file_path)
            else:
                detected_image = Image.open(file_path)

            detected_text = pytesseract.image_to_string(detected_image)

            return render_template('result.html', text=detected_text, image_filename=filename)

    return render_template('upload.html')

@app.route('/detected/<filename>')
def detected_file(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

