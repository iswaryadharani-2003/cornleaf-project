from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

app.secret_key = 'abcdefghijklmnopqrstuvwxyz1234567890'  # Change this to a strong secret key

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        name TEXT ,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()
init_db()

# Load trained model
MODEL_PATH = './model.keras'  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Image prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = round(predictions[0][class_idx] * 100, 2)
    return CLASS_LABELS[class_idx], confidence

# Routes
@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('fullname')
        username = request.form.get('email')
        password = request.form.get('password')

        if not name or not username or not password:
            return render_template('register.html', error='All fields are required')

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (name, username, password) VALUES (?, ?, ?)', 
                           (name, username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        label, confidence = model_predict(file_path, model)
        return render_template('index.html', filename=filename, label=label, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
