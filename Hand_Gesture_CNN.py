# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:56:52 2020

@author: Swapnil Sangal
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

height = 120
width = 320

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip=True)

training_set = train_datagen.flow_from_directory('Datasets/Dataset_2/training_set',
                                                 target_size = (height, width),
                                                 batch_size = 32,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Dataset_2/test_set',
                                            target_size = (height, width),
                                            batch_size = 32,
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')


# CNN Architecture
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", input_shape=[height, width, 1]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))


cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=7, activation='softmax'))


#Compile Model
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fit Model
cnn.fit_generator(training_set,
                  epochs = 25,
                  validation_data = test_set)
