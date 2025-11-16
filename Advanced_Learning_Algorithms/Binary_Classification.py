import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
# Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
#The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
#Each training example becomes a single row in our data matrix X.
#!!!! This gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image. !!!!# 

#The second part of the training set is a 1000 x 1 dimensional vector y that contains labels for the training set
#y = 0 if the image is of the digit 0, y = 1 if the image is of the digit 1.


# Load your file
path = "Mnist_Binary_0_vs_1_1000_20x20.csv"
data = pd.read_csv(path)

# Convert to numpy
data = data.values

# Split into X and y
X = data[:, :-1]          # all rows, all columns except last
y = data[:, -1].reshape(-1, 1)  # last column, reshape to (m,1)

print ('The first element of X is: ', X[0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))    

m, n = X.shape

##### Model Definition, Compilation, and Training #####
"""
 For 20 rounds (epochs):
 1. The model takes all examples in X
 2. Predicts outputs using the current weights
 3. Computes the loss by comparing predictions vs. the true labels y
 4. Uses the optimizer (Adam) to update the weights to reduce the loss
 5. Repeats this entire process for 20 complete passes through the dataset#
 
"""

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    # input layer with 400 features
        tf.keras.layers.Dense(25, activation="sigmoid"), # hidden layer with 25 neurons and sigmoid activation
        tf.keras.layers.Dense(15, activation="sigmoid"), # hidden layer with 15 neurons and sigmoid activation
        tf.keras.layers.Dense(1, activation="sigmoid"),  # output layer with 1 neuron and sigmoid activation
    ], name = "my_model" 
)

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), # What is binary crossentropy? It measures the difference between two probability distributions - the true labels and the predicted probabilities. Then, it computes the average loss across all examples.Then, during training, the model tries to minimize this loss by adjusting its weights.
    optimizer=tf.keras.optimizers.Adam(0.001), # Adam is an optimization algorithm that adjusts the weights of the model based on the computed gradients to minimize the loss function.
)
model.fit(
    X,y,
    epochs=20 # number of complete passes through the training dataset
)


##### Visualizing Predictions #####
""" # For a random 64 examples, feed one example into the model for prediction.
    # Reshape to (1, 400) because Keras requires a 2D input (batch_size × features). 
    # Currently, we have a single example with 400 features (i.e X[0] has 400 features) 
    # We need to change its shape to (1, 400) (where 1 is the batch size) to be accepted by the model.
    # The model performs a forward pass through all layers and outputs a sigmoid value.
    # The sigmoid activation converts the output into a probability between 0 and 1,
    # representing the model’s confidence that the digit is "1".
    # verbose=0 disables progress bar output during prediction. """
 
    
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1)

for i, ax in enumerate(axes.flat): # 8x8 grid for 64 images is flattened to 64 iterations
    idx = np.random.randint(m)

    # reshape image
    img = X[idx].reshape(20, 20).T # Take the idx-th row from X (a 400-dimensional vector) and reshape it to 20x20 for visualization
    ax.imshow(img, cmap='gray')
    
    # prediction
    pred = model.predict(X[idx].reshape(1, 400), verbose=0) #
    if pred >= 0.5:
        yhat = 1
    else:
        yhat = 0

    # title: true, predicted
    ax.set_title(f"{y[idx,0]}, {yhat}", fontsize=8)
    ax.set_axis_off()

fig.suptitle("True Label (y), Predicted Label (ŷ)", fontsize=16)
plt.show()


