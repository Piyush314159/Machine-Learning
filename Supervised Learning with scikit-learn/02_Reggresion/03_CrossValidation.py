from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Load diabetes dataset
diabetes_df = pd.read_csv("Supervised Learning with scikit-learn/Data_set/diabetes_clean.csv")

# Prepare features (X) and target (y)
X = diabetes_df.drop("glucose", axis=1).values  # All columns except glucose
y = diabetes_df["glucose"].values  # Only glucose column

# Set up 6-fold cross-validation
kf = KFold(n_splits=6, shuffle=True, random_state=42)#SEED=42

# Create linear regression model
reg = LinearRegression()

# Perform cross-validation (trains model 6 times, each time testing on different fold) and returns R-square
cv_result = cross_val_score(reg, X, y, cv=kf)

# Print individual scores from each fold
print(cv_result)

# Print average score and standard deviation
print(np.mean(cv_result), np.std(cv_result))

print(np.quantile(cv_result,[0.025,0.975]))#It shows you the range where 95% of your scores fall, helping you understand the variability in model performance across different data splits.