import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load data
homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# 2. Fit linear regression on all 414 rows
model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (per 100m): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# 3. Predict price at 500m (distance_100m = 5)
X_500m = pd.DataFrame({"distance_100m": [5.0]})
pred_500m = model.predict(X_500m)[0]
print(f"Predicted price at 500m: {pred_500m:.2f} price units")

# 4. Plot observations and fitted line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.5, label="Observations", s=20)
x_line = np.linspace(X["distance_100m"].min(), X["distance_100m"].max(), 100)
y_line = model.predict(pd.DataFrame({"distance_100m": x_line}))
plt.plot(x_line, y_line, color="red", linewidth=2, label="Fitted line")
plt.xlabel("Distance to MRT (100m units)")
plt.ylabel("House price of unit area")
plt.title("Task 1: Distance vs House Price")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=150)
plt.show()

# 5. Interpretation
print(f"\nInterpretation: For every additional 100m in distance, "
      f"the house price per unit area changes by {slope:.2f} price units.")