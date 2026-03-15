import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

diabetes_df = pd.read_csv("Supervised Learning with scikit-learn/Data_set/diabetes_clean.csv")

X = diabetes_df.drop("glucose", axis=1).values #taining data as numpy array
y = diabetes_df["glucose"].values# target data as numpy array

#making predictions from a single feature
X_bmi = X[:,3].reshape(-1,1) #resahped feature bcz we need 2-D array(Nx2) for scikit learn

#plotting glucose vs bmi
plt.scatter(X_bmi,y)
plt.ylabel("Blood Glucose Level(mg/dl)")
plt.xlabel("Body Mass Index")

#fitting regression model
reg = LinearRegression()
reg.fit(X_bmi,y)
predictions = reg.predict(X_bmi)#this is basically y we are predicting
plt.plot(X_bmi,predictions, color = "black")
plt.ylabel("Blood Glucose Level(mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
