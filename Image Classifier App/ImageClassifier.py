# Code starts
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Printing stuff
import matplotlib.pyplot as plt

# Load predefined dataset
fashion_mnist = keras.datasets.fashion_mnist

## Pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

## Show data
# print(train_labels[0])
# print(train_images[0])
# plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
# plt.show()

## Define Neural Network structure
model = keras.Sequential([
    # input layer: is a 28x28 image ("Flatten" turns the 28x28 matrix composing the image in a single column vector 781x1) [781 nodes]
    keras.layers.Flatten(input_shape=(28,28)),

    # hidden layer: is 128 nodes. relu returns the value or 0 (works good enough, much faster)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output layer: is 0-9 [10 nodes]. Depending on what piece of clothing return one number (the one with maximum ponderation)
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

## Compile our model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## Train our model, using our training data
model.fit(train_images, train_labels, epochs=5)

## Test our model using our testing data
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(test_images[1], cmap='gray', vmin=0, vmax=255)
plt.show()
print(test_labels[1])

## Make predictions
predictions = model.predict(test_images)
print(predictions[1])
print(list(predictions[1]).index(max(predictions[1])))

print("Code Completed")