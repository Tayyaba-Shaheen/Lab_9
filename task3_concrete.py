import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load data and remove duplicates
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

print(f"Rows after duplicate removal: {len(concrete)}")

# 2. Hold-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

# 3. Linear baseline (same folds)
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_model = LinearRegression()
linear_rmse = -cross_val_score(linear_model, X_train, y_train, cv=folds, scoring=scoring)
print(f"\nLinear baseline mean CV RMSE: {linear_rmse.mean():.2f} MPa")
print("Linear fold RMSEs:", linear_rmse.round(2))

# 4. Ridge pipeline + grid search
ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])

param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}

search = GridSearchCV(
    ridge_pipeline, param_grid, cv=folds, scoring=scoring, refit=True, n_jobs=-1
)
search.fit(X_train, y_train)

# 5. Top 3 settings by rank_test_score
results = search.cv_results_
ranked_idx = np.argsort(results["rank_test_score"])
print("\nTop 3 settings:")
for i in ranked_idx[:3]:
    mean_rmse = -results["mean_test_score"][i]
    params = results["params"][i]
    print(f"  {params} -> mean CV RMSE = {mean_rmse:.2f} MPa")

print(f"\nBest params: {search.best_params_}")
print(f"Best mean CV RMSE: {-search.best_score_:.2f} MPa")

# 6. Test set evaluation
test_pred = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)
print(f"\nTest RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")

# 7. Actual vs predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_pred, alpha=0.6, edgecolors="k", s=30)
min_val = min(y_test.min(), test_pred.min())
max_val = max(y_test.max(), test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y = x")
plt.xlabel("Actual strength (MPa)")
plt.ylabel("Predicted strength (MPa)")
plt.title("Task 3: Concrete Strength – Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=150)
plt.show()

# 8. Comparison and explanation
print(f"\nComparison: Linear baseline CV RMSE = {linear_rmse.mean():.2f} MPa, "
      f"Tuned Ridge CV RMSE = {-search.best_score_:.2f} MPa")