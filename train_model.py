import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob

# Load metadata
metadata = pd.read_csv('/home/kang/Downloads/datasets/archive/HAM10000_metadata.csv')

# Preprocessing
image_size = (128, 128)
batch_size = 32

# Create a dictionary to map image IDs to file paths
image_dirs = ['/home/kang/Downloads/datasets/archive/HAM10000_images_part_1', '/home/kang/Downloads/datasets/archive/HAM10000_images_part_2']
image_paths = {}
for image_dir in image_dirs:
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_id = os.path.splitext(file)[0]
                image_paths[image_id] = os.path.join(root, file)

# Check if the image_paths dictionary is populated
if not image_paths:
    raise ValueError("No images found in the directories. Check the 'image_dirs' and file extensions.")

# Add file paths to metadata
metadata['path'] = metadata['image_id'].map(image_paths.get)

# Handle missing or invalid paths
metadata = metadata.dropna(subset=['path'])  # Drop rows with missing paths
metadata = metadata[metadata['path'].apply(lambda x: os.path.exists(x))]  # Drop rows with invalid paths

# Check if the DataFrame is empty
if metadata.empty:
    raise ValueError("The DataFrame is empty after preprocessing. Check the 'path' column and image directories.")

# Split data into train, validation, and test sets
train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='dx',  # 'dx' is the diagnosis column
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='dx',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='dx',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes in HAM10000
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size,
    epochs=20
)

# Save the model
model.save('skin_cancer_model.h5')