
"""
The application of CNN for image classification using Tensorflow

"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2 # openCV
from sklearn.model_selection import train_test_split

# Names of the classes - will be identified by their indeces
flowerClasses = os.listdir("flowers/")

inputs = [] # images
outputs = [] # expected/real labels of images

height = 100 # new height of all images
width = 100 # new width of all images

# loads and correctly resizes all images
def loadImages(index):
	for image in os.listdir("flowers/" + flowerClasses[index] + "/"):

		image = cv2.imread("flowers/" + flowerClasses[index] + "/" + image)
		image = cv2.resize(image, (height, width))

		inputs.append(np.array(image))
		outputs.append(index)

i = 0
while i < len(flowerClasses):
	loadImages(i)
	i = i + 1

# map outputs to hot one encodings
outputs = keras.utils.to_categorical(outputs, len(flowerClasses))

# divide data to training and testing sets - 30% of data for testing
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.3)

# create model
model = keras.models.Sequential()

# create input layer of the image size
model.add(keras.layers.InputLayer(input_shape = [height, width, 3]))
'''
2D convolution layer:
This layers creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.

filters: the dimensionality of the output space (i.e. the number of output filters in the convolution).
kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
strides: specifying the strides of the convolution along the height and width
'''
model.add(keras.layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2, padding = 'same'))

model.add(keras.layers.Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2, padding = 'same'))

model.add(keras.layers.Conv2D(filters = 96, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2, padding = 'same'))

model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = 2, padding = 'same'))

#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation = 'sigmoid')) # or sigmoid ???
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(flowerClasses), activation = 'softmax'))

optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x = np.array(x_train), y = np.array(y_train), epochs = 50) #
model.summary()


# TEST network:
res = model.evaluate(np.array(x_test), np.array(y_test))
print ("Test loss = " + str(res[0]))
print ("Test acc = " + str(res[1]))