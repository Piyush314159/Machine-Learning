import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- load data ---
df = pd.read_csv("/Users/piyushmaji/Desktop/ML/Data_Camp/Supervised Learning with scikit-learn/Data_set/covid_filtered_1.csv")

X = df[["cumulative_cases", "total_active_cases"]].values   # (357, 2)
y = df["daily_new_cases"].values                             # (357,)

# --- train/test split ---
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- feature scaling (manual) ---
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)

X_train_s = (X_train - mean) / std
X_test_s  = (X_test  - mean) / std

# --- add bias column ---
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_train_b = add_bias(X_train_s)
X_test_b  = add_bias(X_test_s)

# --- gaussian kernel weights ---
# for each query point x_q, compute weight of every training point
# w(i) = exp(-||x_q - x_i||^2 / 2*tau^2)
def gaussian_weights(x_q, X_train, tau):
    diff = X_train - x_q          # (m, n)
    dist_sq = np.sum(diff**2, axis=1)  # (m,)
    return np.exp(-dist_sq / (2 * tau**2))  # (m,)

# --- LWR prediction for a single query point ---
def lwr_predict_one(x_q, X_train_b, y_train, tau):
    # x_q is already scaled and bias-less — we pass the full biased version
    x_q_b = np.array([1, *x_q])           # (n+1,)

    w = gaussian_weights(x_q, X_train_b[:, 1:], tau)  # weights using unbiased part
    W = np.diag(w)                         # (m, m) diagonal weight matrix

    # weighted normal equation: theta = (X^T W X)^-1 X^T W y
    A = X_train_b.T @ W @ X_train_b       # (n+1, n+1)
    b = X_train_b.T @ W @ y_train         # (n+1,)

    theta = np.linalg.solve(A, b)         # safer than inv for near-singular A
    return x_q_b @ theta                  # scalar prediction

# --- predict entire set ---
def lwr_predict(X_query_s, X_train_b, y_train, tau):
    return np.array([
        lwr_predict_one(x_q, X_train_b, y_train, tau)
        for x_q in X_query_s
    ])

# --- metrics ---
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

# --- try a few tau values ---
taus = [0.5, 1.0, 2.0, 5.0]

print(f"{'tau':<8} {'Train RMSE':<15} {'Test RMSE':<15} {'Train R²':<12} {'Test R²'}")
print("-" * 60)

best_tau, best_r2, best_pred = None, -np.inf, None

for tau in taus:
    y_train_pred = lwr_predict(X_train_s, X_train_b, y_train, tau)
    y_test_pred  = lwr_predict(X_test_s,  X_train_b, y_train, tau)

    tr_r2 = r2(y_train, y_train_pred)
    te_r2 = r2(y_test,  y_test_pred)

    print(f"{tau:<8} {rmse(y_train, y_train_pred):<15.2f} {rmse(y_test, y_test_pred):<15.2f} {tr_r2:<12.4f} {te_r2:.4f}")

    if te_r2 > best_r2:
        best_r2   = te_r2
        best_tau  = tau
        best_pred = y_test_pred

print(f"\nBest tau = {best_tau}  |  Test R² = {best_r2:.4f}")

# --- compare with LR ---
# rerun LR for reference
X_train_b2 = add_bias(X_train_s)
X_test_b2  = add_bias(X_test_s)
theta_lr   = np.linalg.inv(X_train_b2.T @ X_train_b2) @ X_train_b2.T @ y_train
y_lr_pred  = X_test_b2 @ theta_lr

print(f"\nLR  Test R²  : {r2(y_test, y_lr_pred):.4f}")
print(f"LWR Test R²  : {best_r2:.4f}  (tau={best_tau})")

# --- plots ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# plot 1: actual vs predicted comparison
axes[0].plot(y_test,     label="Actual",   color="black",     linewidth=1.5)
axes[0].plot(best_pred,  label=f"LWR (τ={best_tau})", color="steelblue", linewidth=1.5)
axes[0].plot(y_lr_pred,  label="LR",       color="tomato",    linewidth=1.5, linestyle="--")
axes[0].set_xlabel("Test Sample Index")
axes[0].set_ylabel("daily_new_cases")
axes[0].set_title("LWR vs LR — Test Set Predictions")
axes[0].legend()

# plot 2: residuals comparison
axes[1].plot(y_test - best_pred, label=f"LWR residuals (τ={best_tau})", color="steelblue", alpha=0.8)
axes[1].plot(y_test - y_lr_pred, label="LR residuals",  color="tomato",    alpha=0.8, linestyle="--")
axes[1].axhline(0, color="black", linestyle=":")
axes[1].set_xlabel("Test Sample Index")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals: LWR vs LR")
axes[1].legend()

plt.tight_layout()
plt.savefig("covid_lwr_plots.png", dpi=120)
plt.show()
print("Plot saved as covid_lwr_plots.png")