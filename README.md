# 🌽 Corn Leaf Disease Detection

## 📖 Project Description

Corn Leaf Disease Detection is a deep learning-based web application that helps farmers and agricultural researchers quickly identify diseases in corn plants through uploaded leaf images.  
The model is trained to classify images into **four categories** using a Convolutional Neural Network (CNN).

**Project Flow:**

- Users register on the platform.
- Users upload a picture of a corn leaf.
- The system processes the image and predicts the type of disease.
- The result is displayed to the user.

The model is trained with **over 4150+ images** divided into 4 disease classes.

---

## 💻 Technologies Used

- **Python** (for model building and backend)
- **TensorFlow** (for deep learning CNN model)
- **OpenCV** (for image processing)
- **HTML5** (for frontend structure)
- **CSS3** (for frontend styling)
- **JavaScript** (for client-side interactivity)

---

## ⚙️ Libraries and Tools

- TensorFlow & Keras (Model development)
- NumPy (Numerical operations)
- OpenCV (Image loading and display)
- Matplotlib (Image visualization)
- ImageDataGenerator (Data augmentation)

---

## 🧠 Algorithm Used

**Convolutional Neural Network (CNN)**

- **Input**: 128×128 color images (RGB)
- **Model Architecture**:

  1. **Convolution Layer 1** → ReLU activation
  2. **MaxPooling Layer 1**
  3. **Convolution Layer 2** → ReLU activation
  4. **MaxPooling Layer 2**
  5. **Flattening** → Converts matrix to vector
  6. **Dense Layer (128 units)** → ReLU activation
  7. **Output Layer (4 units)** → Softmax activation (for 4 classes)

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

- **Training Details**:
  - **Epochs**: 60
  - **Batch Size**: 6 (training), 3 (validation)
  - **Data Augmentation**: Applied (rescaling, zooming, flipping)

---

## 📁 Dataset Information

- **Total Images**: 4150+
- **Classes**:
  - Blight
  - Common Rust
  - Gray Leaf Spot
  - Healthy

Images are divided into **training** and **testing** datasets.

---

## 🔥 Project Workflow

```plaintext
[User Registration] → [Image Upload] → [Image Preprocessing] → [Disease Prediction] → [Result Display]
```

---

## 🚀 How to Use

### Step 1: Dataset Setup

Arrange your dataset in the following folder structure:

```plaintext
./root_folder/
    └── dataset/
        ├── class_1/
        ├── class_2/
        ├── class_3/
        └── class_N/
```

Each `class_x/` folder should contain images related to that class.

> Example classes: `Blight/`, `Common Rust/`, `Gray Leaf Spot/`, `Healthy/`

---

### Step 2: Install Dependencies

Make sure you have all required Python libraries installed. Run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```plaintext
tensorflow
opencv-python
numpy
matplotlib
flask
```

---

### Step 3: Train the Model

Open your terminal and run the following command to start model training:

```bash
python model/train.py
```

This will:

- Preprocess the dataset
- Train the CNN model
- Save the model files (`model.h5`, `model.keras`, and `model1.json`)

---

### Step 4: Run the Application

After successful model training, start the web app:

```bash
python app.py
```

The application will launch, allowing users to upload corn leaf images and detect diseases!

---

> ✨ **Note**: Make sure all required Python libraries are installed before running the training or app scripts.

---

## 📷 Screenshots (Optional)

_(Add your app screenshots here if you have)_

---

## 📢 Future Enhancements

- Improve model accuracy by using larger datasets.
- Add email alerts or reports for detected diseases.
- Add mobile responsiveness to the frontend UI.

---

## 🤝 Contributions

Feel free to fork this project, make improvements, and submit pull requests!

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
