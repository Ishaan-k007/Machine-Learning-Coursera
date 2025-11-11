import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
#Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
#The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
#Each training example becomes a single row in our data matrix X.

#!!!! This gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image. !!!!# 


#The second part of the training set is a 1000 x 1 dimensional vector y that contains labels for the training set
#y = 0 if the image is of the digit 0, y = 1 if the image is of the digit 1.

X,y = np.load("X.npy"), np.load("y.npy")
print ('The first element of X is: ', X[0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])

print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))    


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

#In the cell below, the code randomly selects 64 rows from X, maps each row back to a 20 pixel by 20 pixel grayscale image and displays the images together.
#The label for each image is displayed above the image.

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        tf.keras.layers.Dense(25, activation="sigmoid"),
        tf.keras.layers.Dense(15, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name = "my_model" 
)

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)


model.fit(
    X,y,
    epochs=20
)

# X has shape (1000, 400), so X[0] selects one example, which has shape (400,) 
# Neural networks expect inputs in the form (batch_size, number_of_features),
# so we reshape the single example to shape (1, 400) to add the batch dimension.
# model.predict then performs forward propagation and outputs the probability
# that this image is the digit '1'. 
prediction = model.predict(X[0].reshape(1, 400))
print(prediction) 


if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
    
    # Display the label above the image
ax.set_title(f"{y[random_index,0]},{yhat}")
ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()

