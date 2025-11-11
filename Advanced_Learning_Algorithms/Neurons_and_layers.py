import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

# Using the Example from Course 1 -> linear regression with one neuron

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

# A Function implemented by a neuron with no activation function is:
# f_wb = W*X + b

# Defining a layer with one neuron and compare it to linear regression model

linear_layer = tf.keras.layers.Dense(units=1, input_shape=(1,), activation='linear')
linear_layer.get_weights()  # Initial weights and bias

# instantiate the layer
a1 = linear_layer(X_train[0].reshape(1,1)) # reshape due to input shape requirement to make it 2D. What does reshape mean?
print(a1)


set_w = np.array([[200]])
set_b = np.array([100])
# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print(alin)


prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b