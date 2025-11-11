import numpy as np
def compute_cost(X, y, w, b):
    """
    Compute cost for logistic regression
    
    Args:
        X (ndarray): Shape (m, n), feature matrix
        y (ndarray): Shape (m, 1), labels (0 or 1)
        w (ndarray): Shape (n, 1), weights
        b (float): bias
    
    Returns:
        cost (float): The cost value
    """
    m = X.shape[0]   # number of training examples

    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b         # w*x + b
        h = 1 / (1 + np.exp(-z))        # sigmoid(z)
        cost += - (y[i] * np.log(h) + (1 - y[i]) * np.log(1 - h))

    cost = cost / m                    # average over all examples
    return cost