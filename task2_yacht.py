import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

# 1. Load data (headerless, whitespace-separated)
yachts = pd.read_csv("yacht_hydrodynamics.data", sep=r"\s+", header=None)
X = yachts[[5]]   # Froude number
y = yachts[6]     # Residual resistance per unit displacement

# 2. Build models
linear_model = LinearRegression()
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression()),
])

# 3. Same 5-fold shuffled CV
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(linear_model, X, y, cv=folds, scoring=scoring)
quadratic_rmse = -cross_val_score(quadratic_model, X, y, cv=folds, scoring=scoring)

# 4. Report
print("Linear fold RMSEs:", linear_rmse.round(2))
print(f"Linear mean RMSE: {linear_rmse.mean():.2f}")
print("Quadratic fold RMSEs:", quadratic_rmse.round(2))
print(f"Quadratic mean RMSE: {quadratic_rmse.mean():.2f}")

# 5. Choose better model
better = "Quadratic" if quadratic_rmse.mean() < linear_rmse.mean() else "Linear"
print(f"\nBetter model (lower mean CV RMSE): {better}")