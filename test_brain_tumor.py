# -----------------------------
# Brain Tumor Prediction Script
# -----------------------------

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# -----------------------------
# Step 1: Load the trained model
# -----------------------------
model = load_model('brain_tumor_model.h5')
print("Model loaded successfully!")

# -----------------------------
# Step 2: Path to new image
# -----------------------------
img_path = 'new_mri_image.jpg'  # Replace with your image filename

# -----------------------------
# Step 3: Load and preprocess image
# -----------------------------
img = image.load_img(img_path, target_size=(128,128))  # resize to 128x128
img_array = image.img_to_array(img)                    # convert to array
img_array = np.expand_dims(img_array, axis=0)          # add batch dimension
img_array = img_array / 255.0                          # rescale pixel values

# -----------------------------
# Step 4: Make prediction
# -----------------------------
prediction = model.predict(img_array)
class_index = np.argmax(prediction)  # get the index of highest probability

# Mapping class index to label
labels = ['No Tumor', 'Tumor']       # Must match training generator order

print(f"Prediction: {labels[class_index]}")
