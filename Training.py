# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:28:07 2025

@author: amanr
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

DATASET_PATH = r"E:\360_project\Dataset\data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4
HIDDEN_UNITS = [128, 64]
MODEL_SAVE_PATH = r"E:\360_project\Models\final_xception_model.keras"

class_names = ['Grade_A', 'Grade_B', 'Grade_C']

# Data loading function
def load_dataset(path, img_size, batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True  # Shuffling data
    ).map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Augmentation
    return ds.prefetch(tf.data.experimental.AUTOTUNE)

dataset = load_dataset(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

# Split dataset into training and validation sets
val_size = 0.2  # 20% of data for validation
dataset_size = len(dataset)
val_ds = dataset.take(int(val_size * dataset_size))
train_ds = dataset.skip(int(val_size * dataset_size))

# Load Existing Model if Available
if os.path.exists(MODEL_SAVE_PATH):
    model = load_model(MODEL_SAVE_PATH)
    print("Loaded existing model from checkpoint.")
else:
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(HIDDEN_UNITS[0], activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(HIDDEN_UNITS[1], activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    print("Initialized new model.")

model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

# Training
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint, early_stopping, reduce_lr])

# Save model in .keras extension
model.save(MODEL_SAVE_PATH)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

