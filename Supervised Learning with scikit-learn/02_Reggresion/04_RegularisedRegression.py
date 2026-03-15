from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load diabetes dataset
diabetes_df = pd.read_csv("Supervised Learning with scikit-learn/Data_set/diabetes_clean.csv")

# Prepare features (X) and target (y)
X = diabetes_df.drop("glucose", axis=1).values  # All columns except glucose
y = diabetes_df["glucose"].values  
names=diabetes_df.drop("glucose",axis=1).columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


#Ridge
#===========
from sklearn.linear_model import Ridge
scores_ridge=[]
for a in [0.1,1.0,10.0,100.0,100.0]:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train,y_train)
    y_pred_ridge = ridge.predict(X_test)
    scores_ridge.append(ridge.score(X_test,y_test))

print(scores_ridge)

#feature selection using ridge
#===============================
ridge1 = Ridge(alpha=0.1)
ridge1_coef = ridge1.fit(X,y).coef_

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.bar(names,ridge1_coef,color="RED")
plt.title("Ridge")
plt.xticks(rotation=45)


#Lasso
#=============
from sklearn.linear_model import Lasso
scores_lasso=[]
for a in[0.01,1.0,5.0,10.0,20.0,50.0]:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train,y_train)
    y_pred_lasso = lasso.predict(X_test)
    scores_lasso.append(lasso.score(X_test,y_test))

print(scores_lasso)

#Feature selection using Lasso
#======================================
lasso1 = Lasso(alpha=0.1)
lasso1_coef = lasso1.fit(X,y).coef_

plt.subplot(1,2,2)
plt.bar(names,lasso1_coef)
plt.title("Lasso")
plt.xticks(rotation=45)
plt.show()
