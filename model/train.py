from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

# ❌ REMOVE: tf.compat.v1.disable_eager_execution()

# ✅ Fix: Define CNN with correct output classes
classifier = Sequential()

# Convolution + Pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolution + Pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully Connected Layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=4, activation='softmax'))  # ✅ Changed from 10 to 4

# Compile the Model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r'D:\corn leaf disease dL\dataset\data', target_size=(128, 128), batch_size=6, class_mode='categorical'
)

valid_set = test_datagen.flow_from_directory(
    r'D:\corn leaf disease dL\dataset\test', target_size=(128, 128), batch_size=3, class_mode='categorical'
)

labels = training_set.class_indices
print(labels)

# Train the Model
classifier.fit(training_set, steps_per_epoch=20, epochs=60, validation_data=valid_set)

# Save Model
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

# classifier.save_weights("my_model_weights.weights.h5")
classifier.save("model.h5")
classifier.save("model.keras")

print("Saved model to disk")

# Image Prediction
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\corn leaf disease dL\dataset\data\Blight\Corn_Blight (1).jpeg')  # Update the correct image path
img_resize = cv2.resize(img, (128, 128))

# Convert BGR to RGB
b, g, r = cv2.split(img_resize)
rgb_img = cv2.merge([r, g, b])

plt.imshow(rgb_img)
label_map = training_set.class_indices

# Prepare image for prediction
img_rank4 = np.expand_dims(rgb_img / 255, axis=0)

# ✅ Fix: Use np.argmax instead of removed `predict_classes()`
prediction = np.argmax(classifier.predict(img_rank4), axis=-1)
h = list(label_map.keys())[prediction[0]]

# Display Prediction
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(img, h, (10, 30), font, 1.0, (0, 0, 255), 1)
cv2.imshow(h, img)

print(h)
