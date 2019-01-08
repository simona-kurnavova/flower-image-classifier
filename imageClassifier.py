
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

epochs_count = 30

# Names of the classes - will be identified by their indeces
flowerClasses = os.listdir("flowers/")

inputs = [] # images
outputs = [] # expected/real labels of images
img_names = [] # file names of the images

height = 80 # new height of all images
width = 80 # new width of all images

# loads and correctly resizes all images
def loadImages(index):
	for image in os.listdir("flowers/" + flowerClasses[index] + "/"):
		image_path = "flowers/" + flowerClasses[index] + "/" + image
		image = cv2.imread(image_path)
		image = cv2.resize(image, (height, width))
		inputs.append(np.array(image))
		outputs.append(index)
		img_names.append(image_path)

i = 0
while i < len(flowerClasses):
	loadImages(i)
	i = i + 1

# divide data to training and testing sets - 30% of data for testing
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.2, random_state = 13)

# map train outputs to hot one encodings (categorical)
y_train = keras.utils.to_categorical(y_train, len(flowerClasses))

# create model
model = keras.models.Sequential()

# create input layer of the image size
model.add(keras.layers.InputLayer(input_shape = [height, width, 3]))

#2D convolution layers, pooling layers, normalization, dropout
model.add(keras.layers.Conv2D(filters = 128, kernel_size = 5, padding='same', strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, padding='same', strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, padding='same', strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 1024, kernel_size = 3, padding='same', strides = 2, activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation = 'relu'))
model.add(keras.layers.Dense(512, activation = 'relu'))
model.add(keras.layers.Dropout(0.2))	
model.add(keras.layers.Dense(len(flowerClasses), activation = 'softmax'))

model.compile(optimizer = "adadelta", loss = 'categorical_crossentropy', metrics=['accuracy'])

for i in range(epochs_count):
	history = model.fit(x = np.array(x_train), y = np.array(y_train), epochs = 1) #batch_size = 128 

	# loss and accuracy for testing data
	y_test_categorical = keras.utils.to_categorical(y_test, len(flowerClasses))
	res = model.evaluate(np.array(x_test), np.array(y_test_categorical))
	print ("Test loss = " + str(res[0]))
	print ("Test acc = " + str(res[1]))

model.summary()

# TESTING OF THE NETWORK:
# Generate confusion matrix for testing data
predictions = model.predict_classes(np.array(x_test))
matrix = tf.confusion_matrix(labels = y_test, predictions = predictions)
sess = tf.Session()
with sess.as_default():
	print(sess.run(matrix)) # prints matrix

# find all wrongly classified patterns in the data
predict = model.predict_classes(np.array(inputs))
for i in range(len(predict)):
	if predict[i] != outputs[i]:
		print(predict[i], '==>', outputs[i], ' ', img_names[i])
