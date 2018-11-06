# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
import PIL
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
	'/home/sonu/Desktop/data/test',
	target_size=(256, 256),
	batch_size=10,
	class_mode='binary')

train_generator = train_datagen.flow_from_directory(
        '/home/sonu/Desktop/data/train',
        target_size=(256, 256),
        batch_size=10,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/sonu/Desktop/data/test',
        target_size=(256, 256),
        batch_size=10,
        class_mode='binary')


# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(20, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 100 nodes
model.add(Flatten())
model.add(Dense(100, activation="relu"))
# Hidden layer with 10 nodes
model.add(Dense(10, activation="relu"))

# Output layer with 2 nodes (one for each class: thumbs up/thumbs down
model.add(Dense(2, activation="softmax"))

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.fit(train_images, train_labels, epochs=10)
model.fit_generator(
        train_generator,
        steps_per_epoch=45,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=80)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
