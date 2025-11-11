import numpy as np
import matplotlib.pyplot as plt


# X is an input numpy array which has 6 training examples each with 2 features
# y is an output numpy array which has 6 outputs (0 or 1) for each training example 
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

# Plotting function to visualize the data points
plt.figure(figsize=(4,4))              # Create new figure
plt.scatter(X[:,0], X[:,1], c=y.flatten())     # Plot all points
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='b', marker='x', label='y=1') 
plt.legend()
plt.axis([0, 4, 0, 3.5])
plt.set_ylabel = ('$x_1$')
plt.set_xlabel = ('$x_0$')
plt.show()


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar or numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g = 1/(1+np.exp(-z))
    return g


# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

plt.figure(figsize=(5,3))           # Create new figure
plt.plot(z, sigmoid(z), c="b")      # Plot z vs sigmoid(z)
plt.title("Sigmoid function")
plt.ylabel('sigmoid(z)')
plt.xlabel('z')
plt.show()


# Let's say that you trained the model and got the parameters as follows:
# f_wb = w0*x0 + w1*x1 + b
#   where w0 = 1, w1 = 1, b = -3
# Thus the decision boundary is defined by:
#   z = 0
#   0 = 1*x0 + 1*x1 - 3
#   x1 = 3 - x0

# Choose values between 0 and 6 for plotting the boundary line
x0 = np.arange(0,6)

# Compute corresponding x1 values (based on the decision boundary equation)
x1 = 3 - x0

plt.figure(figsize=(5,4))            # Create new figure
plt.plot(x0, x1, c="r")              # Plot the decision boundary
plt.axis([0, 4, 0, 3.5])

# Fill the region below the decision boundary line
plt.fill_between(x0, x1, y2=0, alpha=0.2)

# Plot the original data again
plt.scatter(X[:,0], X[:,1], c=y.flatten())

plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.show()