# Multivariate Linear Regression Example
# Steps to implement:
# 1) Create a dataset with multiple features
# 2) Create a function to calculate fx_wb
# 3) Create a function to compute the cost
# 4) Create a function to compute the partial derivatives
# 5) Create a function to perform gradient descent
# 6) Run the algorithm and plot the cost vs iterations

import copy, math
import numpy as np
import matplotlib.pyplot as plt



# Example training data - shown as (size, bedrooms, floors, age) -> price 
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) 
y_train = np.array([460, 232, 178])

# The parameters are stored as a numpy matrix - IMPORTANT as you can then preform dot products on them which is very fast!
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# The multivariate linear regression model is:
# f_wb = w1x1 + w2x2 + w3x3 -> it is important to also represent this in vector notation
# Initially it is loaded with selected initial values

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predicted_fwb(x, w, b):
    """
    Predict the output for a single instance x using the linear model with parameters w and b.
    This is much faster than looping over each feature in x and summing the result.
    
    Args:
    x : (n,) numpy array - Input features for a single instance
    w : (n,) numpy array - Weights for each feature
    b : float - Bias term
    
    Returns:
    float - Predicted output
    """
    # Compute the dot product between input features and weights, then add the bias term
    prediction = np.dot(x, w) + b
    return prediction

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predicted_fwb(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

def compute_cost(x, y, w, b):
    m = x.shape[0]  # number of training examples
    total_cost = 0.0
    
    
    for i in range(m):
        f_wb = predicted_fwb(x[i,:], w, b)  # Predicted value for row i of x train dotted with w and add b
        print("T1", x[i,:])
        total_cost += (f_wb - y[i]) ** 2  # Squared error
    
    total_cost = total_cost / (2 * m)  # Average cost
    return total_cost

# Compute and display the cost with initial parameters
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"Initial cost: {cost}")

def compute_partial_derivative(x, y, w, b):
    """
Computes the gradient of the cost function J with respect to the weight vector w
for a linear regression model with two training examples and two features.

Mathematically:

    ∂J/∂w = (1/2) * [
        ( (w1*x11 + w2*x12 + b - y1) * [x11, x12] )
      + ( (w1*x21 + w2*x22 + b - y2) * [x21, x22] )
    ]

where:
    - w1, w2 : weights corresponding to features 1 and 2
    - b      : bias term
    - xij    : value of feature j for example i
    - yi     : true label for example i
    - ŷi     : predicted output for example i = w1*xi1 + w2*xi2 + b

This formula averages the contribution of each training example
to compute how much each weight (w1, w2) should be adjusted.
"""

    m, n = x.shape  # number of training examples and number of features
    dj_dw = np.zeros(n)  # Initialize gradient for weights
    dj_db = 0.0  # Initialize gradient for bias
    
    for i in range(m):
        f_wb = predicted_fwb(x[i,:], w, b)  # Predicted value for row i of x train dotted with w and add b
        error = f_wb - y[i]  # Error term
        
        for j in range(n):
            dj_dw[j] += error * x[i, j]  # Accumulate gradient for each weight
        
        dj_db += error  # Accumulate gradient for bias
    
    dj_dw /= m  # Average gradient for weights
    dj_db /= m  # Average gradient for bias
    
    return dj_dw, dj_db

# Compute and display the partial derivatives with initial parameters
dj_dw, dj_db = compute_partial_derivative(X_train, y_train, w_init, b_init)
print(f"Initial dj_dw: {dj_dw}, Initial dj_db: {dj_db}")

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Perform gradient descent to learn the parameters w and b.
    
    Args:
    x : (m,n) numpy array - Input features for m instances and n features
    y : (m,) numpy array - True output values for m instances
    w_in : (n,) numpy array - Initial weights for each feature
    b_in : float - Initial bias term
    alpha : float - Learning rate
    num_iters : int - Number of iterations for gradient descent
    
    Returns:
    w : (n,) numpy array - Learned weights after gradient descent
    b : float - Learned bias term after gradient descent
    J_history : list - History of cost function values
    p_history : list - History of parameter values (w, b)
    """
    J_history = []
    p_history = []
    b = b_in
    w = copy.deepcopy(w_in)  # Avoid modifying the input weights directly
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using compute_partial_derivative
        dj_dw, dj_db = compute_partial_derivative(x, y, w, b)     

        # Update Parameters
        b -= alpha * dj_db                            
        w -= alpha * dj_dw                            

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion 
            J_history.append(compute_cost(x, y, w, b))
            p_history.append((copy.deepcopy(w), b))
        
        # Print cost every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i}: Cost {J_history[-1]}, w: {w}, b: {b}")

    return w, b, J_history, p_history

# Run gradient descent with initial parameters
alpha = 5.0e-7  # Learning rate
iterations = 10000  # Number of iterations
w_final, b_final, J_hist, p_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iterations)
print(f"Final parameters after gradient descent: w: {w_final}, b: {b_final}")
print(f"Final cost after gradient descent: {J_hist[-1]}") 
# Plot the cost vs iterations
plt.plot(J_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

