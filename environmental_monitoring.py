# Import necessary libraries
import os
import requests
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Dataset Collection: Download and organize COCO dataset
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

url = "http://images.cocodataset.org/zips/train2017.zip"
response = requests.get(url)

with open("data/train/train2017.zip", "wb") as f:
    f.write(response.content)
print("Dataset downloaded successfully.")

# 2. Dataset Analysis: Load and visualize object categories
with open("data/annotations/instances_train2017.json", "r") as f:
    annotations = json.load(f)

categories = [category['name'] for category in annotations['categories']]

plt.figure(figsize=(12, 6))
plt.bar(categories, [category['id'] for category in annotations['categories']])
plt.xticks(rotation=90)
plt.title("Distribution of Object Categories in COCO Dataset")
plt.show()

# 3. Data Preprocessing: Augment and normalize data
datagen = ImageDataGenerator(
    rescale=1.0/255.0, 
    rotation_range=20,   
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    "data/train/",
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical'
)

# 4. Model Training: Split dataset and compile the model
all_images = [os.path.join("data/train", img) for img in os.listdir("data/train")]
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

model = tf.keras.applications.YOLOv4(input_shape=(416, 416, 3), weights='coco')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=None,  
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
)

# 5. Evaluation: Make predictions and calculate precision and recall
predictions = model.predict(train_generator)
precision = precision_score(train_generator.classes, predictions.argmax(axis=1), average='macro')
recall = recall_score(train_generator.classes, predictions.argmax(axis=1), average='macro')
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

# 6. Example Use Case: Object detection on satellite or drone image
image_path = "data/sample_image.jpg"
image = tf.image.decode_jpeg(tf.io.read_file(image_path))
image_resized = tf.image.resize(image, (416, 416)) / 255.0
detections = model.predict(image_resized[None, ...])
print("Detected objects:", detections)
