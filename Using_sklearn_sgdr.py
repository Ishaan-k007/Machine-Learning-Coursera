import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

X_features = ['size', 'bedrooms', 'floors', 'age']

# Standardize the features
# Import StandardScaler from sklearn.preprocessing then scaler.fit_transform(X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(f"Scaled X_train:\n{X_train_scaled}")



sgdr = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)
sgdr.fit(X_train_scaled, y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"Normalized weights: {w_norm}, bias: {b_norm}")

# Predicting a new data point

y_pred = sgdr.predict(X_train_scaled)
print(f"Predictions on training data: {y_pred}")

#Maanually predicting a new data point

y_pred_manual = np.dot(X_train_scaled, w_norm) + b_norm
print(f"Manual predictions on training data: {y_pred_manual}")

# Visualizing the results
# plt.subplots(1,4) -> 4 subplots in a row
# figsize=(12,3) -> size of the figure
# sharey=True -> share the y axis
# fig -> the figure
# ax -> an array of 4 axes e.g. ax[0] (first plot (size v prices)), ax[1], ax[2], ax[3] 
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# X_train[:, i] -> all rows of column i
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='Target') # plot the target values
    ax[i].scatter(X_train[:, i], y_pred, label='Prediction') # plot the predicted values
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target vs Prediction using z-score normalized model")
plt.show()

