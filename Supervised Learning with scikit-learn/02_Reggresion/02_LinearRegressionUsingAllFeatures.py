from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the diabetes dataset from CSV file into a pandas DataFrame
diabetes_df = pd.read_csv("Supervised Learning with scikit-learn/Data_set/diabetes_clean.csv")

# Create feature matrix X by removing the target column 'glucose'
# axis=1 means drop column (axis=0 would drop row)
# .values converts DataFrame to NumPy array for sklearn compatibility
X = diabetes_df.drop("glucose", axis=1).values

# Create target array y containing only the glucose values we want to predict
# .values converts pandas Series to NumPy array
y = diabetes_df["glucose"].values

# Split data into training and testing sets
# 70% of data goes to training (X_train, y_train)
# 30% of data goes to testing (X_test, y_test)
# random_state=42 ensures reproducible splits across different runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Linear Regression model object
# At this point, the model hasn't learned anything yet
reg_all = LinearRegression()

# Train (fit) the model on the training data
# The model learns coefficients (weights) for each feature
# to best predict glucose levels from the input features
reg_all.fit(X_train, y_train)

# Use the trained model to predict glucose values for the test set
# The model has never seen X_test during training
# This tests how well the model generalizes to new data
y_pred = reg_all.predict(X_test)

# Display the array of predicted glucose values
print(f"Prediction of y : {y_pred}")

# Calculate R² (R-squared) score - coefficient of determination
# R² measures how well the model explains variance in the data
# Range: 0 to 1, where 1 = perfect fit, 0 = no better than mean
# Compares predicted values against actual test values
y_r2 = reg_all.score(X_test, y_test)
print(f"R² Score : {y_r2}")
#R² (R-squared) answers the question: "How much of the variation in glucose levels can my model explain?"

# Calculate Root Mean Squared Error (RMSE)
# RMSE measures average prediction error in same units as target (mg/dL)
# Lower RMSE indicates better model performance
# It's the square root of the mean of squared differences between actual and predicted
y_rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE : {y_rmse}")
print(reg_all.coef_)  # Shows [a1, a2, a3, ..., an] for each feature,Coefficients (a1, a2, a3, ...):Magnitude shows how much that feature matters
#Sign shows direction of relationship
#Positive: as feature increases, glucose increases
#Negative: as feature increases, glucose decreases
print(reg_all.intercept_)  # Shows b (the constant term)
#Intercept (b):
#Baseline glucose level when all features = 0
#Shifts the entire prediction line up or down