from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# TensorFlow GPU configuration
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Define CNN model
classifier = Sequential()

# Convolution + Pooling Layer 1
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution + Pooling Layer 2
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully Connected Layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=4, activation='softmax'))  # 4 classes

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
training_set = train_datagen.flow_from_directory(
    r'D:\corn leaf disease dL\dataset\data',
    target_size=(128, 128),
    batch_size=6,
    class_mode='categorical'
)

valid_set = test_datagen.flow_from_directory(
    r'D:\corn leaf disease dL\dataset\test',
    target_size=(128, 128),
    batch_size=3,
    class_mode='categorical'
)

# Show class labels
labels = training_set.class_indices
print("Class Labels:", labels)

# Train the model
history = classifier.fit(
    training_set,
    steps_per_epoch=20,
    epochs=60,
    validation_data=valid_set
)

# Save Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")  # ✅ Save accuracy graph
plt.show()

# Save Loss Graph
plt.figure()
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")  # ✅ Save loss graph
plt.show()

# Save the model
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

classifier.save("model.h5")
classifier.save("model.keras")
print("✅ Model saved successfully.")

# Image Prediction Test
img_path = r'D:\corn leaf disease dL\dataset\data\Blight\Corn_Blight (1).jpeg'  # Update as needed
img = cv2.imread(img_path)
img_resize = cv2.resize(img, (128, 128))

# Convert BGR to RGB
rgb_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title("Input Image")
plt.axis('off')
plt.show()

# Prepare image for prediction
img_rank4 = np.expand_dims(rgb_img / 255.0, axis=0)

# Predict
prediction = np.argmax(classifier.predict(img_rank4), axis=-1)
predicted_label = list(labels.keys())[prediction[0]]

# Display prediction result
print("Predicted Label:", predicted_label)

# Display image with label using OpenCV
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(img, predicted_label, (10, 30), font, 1.0, (0, 0, 255), 1)
cv2.imshow(predicted_label, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
