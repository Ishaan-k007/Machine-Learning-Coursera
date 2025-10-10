# Gradient Descent Algorithm for Linear Regression Manually coded:
#Steps:
# 1) Iterate through the points 
# 2) The gradient descent algorithm changes w,b according to the following equation w:= w - alpha * partial dif f_wb respect to w and b:= b - alpha * partial dif f_wb respect to b 
# 3) In order to do this you need to create a function to calc the partial derivates 
# 4) Then calc the w,b
# 5) The cost with the associated w and b is a nice visual to see how gradient descent works



import numpy as np
x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])




#Function to compute the cost
# predicted model is f_wb = wx + b
# cost = 1/2m sigma(1 to m) (f_wb - y[i])^2
#sigma (1 to m) is represented as a for loop
def compute_cost(x, y, w, b):
    number_of_samples = x.shape[0]
    cost = 0
    for i in range(number_of_samples):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1/(2*number_of_samples) * cost
    
    return total_cost

# formula is calculated by taking the partial derivative of 
def compute_partial_derivative(x, y, w , b):
    number_of_samples = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(number_of_samples):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i
    dj_dw = dj_dw / number_of_samples
    dj_db = dj_db / number_of_samples
    
    return dj_dw, dj_db

#Gradient descent function
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_partial_derivative(x, y, w , b)     

        # Update Parameters
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost(x, y, w , b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations)
#print the results
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    