# -----------------------------
# Brain Tumor Detection Project
# -----------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Prepare image generators
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    r'C:\Users\ziaud\OneDrive\Desktop\BrainTumorDetection\brain_tumor_dataset',          # dataset folder path
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    r'C:\Users\ziaud\OneDrive\Desktop\BrainTumorDetection\brain_tumor_dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# Step 2: Visualize some images
# -----------------------------
images, labels = next(train_gen)
class_labels = list(train_gen.class_indices.keys())

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    label_index = labels[i].argmax()
    plt.title(class_labels[label_index])
    plt.axis('off')
plt.show()

# -----------------------------
# Step 3: Build the CNN model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# Step 4: Train the CNN model
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10      # you can increase this for better accuracy
)

# -----------------------------
# Step 5: Plot training accuracy
# -----------------------------
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------------
# Step 6: Save the trained model
# -----------------------------
model.save('brain_tumor_model.h5')
print("Model saved as brain_tumor_model.h5")
