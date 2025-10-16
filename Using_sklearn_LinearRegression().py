
# ------------------------------------------------------------
# | Feature             | LinearRegression() | SGDRegressor()         |
# |----------------------|--------------------|------------------------|
# | Algorithm            | Normal Equation    | Gradient Descent       |
# | Speed (small data)   | Fast               | Slower                 |
# | Speed (large data)   | Slow               | Scales better          |
# | Feature scaling      | Not needed         | Required               |
# | Regularization       | None               | L1, L2, ElasticNet     |
# | Deterministic        | Yes                | No (random init)       |
# | Accuracy             | Exact              | Approximate            |

# Analogy:
# LinearRegression() is like directly solving for where the ball will land using math.
# SGDRegressor() is like rolling the ball downhill step by step until it reaches the bottom.
# Both can reach the same spot — one does it instantly, the other iteratively.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print("\nLinearRegression model summary:")
print(lin_reg)


b_lin = lin_reg.intercept_
w_lin = lin_reg.coef_

print("\nModel parameters:")
print(f"w: {w_lin}")
print(f"b: {b_lin}")

print("\nModel parameters from SGDRegressor (for comparison):")
print("w: [109.95 -20.97 -32.35 -38.07], b: 363.15")


y_pred_lin = lin_reg.predict(X_train)
y_pred_manual = np.dot(X_train, w_lin) + b_lin


fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='Target')
    ax[i].scatter(X_train[:, i], y_pred_lin, label='Prediction')
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target vs Prediction using LinearRegression model")
plt.show()
