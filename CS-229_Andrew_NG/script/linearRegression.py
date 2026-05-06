import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- load data ---
df = pd.read_csv("/Users/piyushmaji/Desktop/ML/CS-229_Andrew_NG/Data/Advertising.csv")

# features: cumulative_cases, total_active_cases
# target:   daily_new_cases
X = df[["TV","Radio","Newspaper"]].values   # (357, 2)
y = df["Sales"].values                             # (357,)

# --- train/test split (manual, no sklearn) ---
split = int(0.8 * len(y))          # 80% train, 20% test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- feature scaling (manual standardization) ---
mean = X_train.mean(axis=0)        # (2,)  mean of each feature
std  = X_train.std(axis=0)         # (2,)  std of each feature

X_train_s = (X_train - mean) / std
X_test_s  = (X_test  - mean) / std   # use train stats on test

# --- add bias column (column of ones) ---
m_train = X_train_s.shape[0]
m_test  = X_test_s.shape[0]

X_train_b = np.hstack([np.ones((m_train, 1)), X_train_s])  # (285, 3)
X_test_b  = np.hstack([np.ones((m_test,  1)), X_test_s])   # (72,  3)

# --- normal equation: theta = (X^T X)^-1 X^T y ---
theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

print("theta (intercept, coef_cumulative, coef_active):")
print(f"  θ₀ (intercept)          : {theta[0]:.4f}")
print(f"  θ₁ (cumulative_cases)   : {theta[1]:.4f}")
print(f"  θ₂ (total_active_cases) : {theta[2]:.4f}")

# --- predict ---
y_train_pred = X_train_b @ theta
y_test_pred  = X_test_b  @ theta

# --- metrics (manual) ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

print(f"\nTrain MSE  : {mse(y_train, y_train_pred):.2f}")
print(f"Test  MSE  : {mse(y_test,  y_test_pred):.2f}")
print(f"Train RMSE : {np.sqrt(mse(y_train, y_train_pred)):.2f}")
print(f"Test  RMSE : {np.sqrt(mse(y_test,  y_test_pred)):.2f}")
print(f"Train R²   : {r2(y_train, y_train_pred):.4f}")
print(f"Test  R²   : {r2(y_test,  y_test_pred):.4f}")

# --- plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# plot 1: actual vs predicted (test set)
axes[0].scatter(y_test, y_test_pred, alpha=0.6, color="steelblue")
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", label="perfect fit")
axes[0].set_xlabel("Actual daily_new_cases")
axes[0].set_ylabel("Predicted daily_new_cases")
axes[0].set_title("Actual vs Predicted (Test Set)")
axes[0].legend()

# plot 2: residuals
residuals = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuals, alpha=0.6, color="darkorange")
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot (Test Set)")

plt.tight_layout()
plt.savefig("/Users/piyushmaji/Desktop/ML/CS-229_Andrew_NG/Plots/Advertising.png", dpi=120)
plt.show()
print("\nPlot saved as covid_lr_plots.png")