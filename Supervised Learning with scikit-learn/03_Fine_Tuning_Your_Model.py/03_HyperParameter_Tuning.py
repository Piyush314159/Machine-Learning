from sklearn.model_selection import KFold,GridSearchCV,train_test_split,RandomizedSearchCV
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

# Load diabetes dataset
diabetes_df = pd.read_csv("/Users/piyushmaji/Desktop/Python/ML/Supervised Learning with scikit-learn/Data_set/diabetes_clean.csv")
# Prepare features (X) and target (y)
X = diabetes_df.drop("glucose", axis=1).values  # All columns except glucose
y = diabetes_df["glucose"].values  # Only glucose column

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21) 

# Set up 6-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)#SEED=42

param_grid = {"alpha" : np.arange(0.0001,1,10),
               "solver" : ["sag", "lsqr"]}#different routes to find distance

ridge = Ridge(max_iter=10000)

#GridSearchCV
#====================
ridge_cv1 = GridSearchCV(ridge, param_grid ,cv=kf)
ridge_cv1.fit(X_train,y_train)
print(ridge_cv1.best_params_,ridge_cv1.best_score_)

#RandomSearchCV
#====================
ridge_cv2 = RandomizedSearchCV(ridge, param_grid , cv=kf, n_iter=2)
ridge_cv2.fit(X_train,y_train)
print(ridge_cv2.best_params_,ridge_cv2.best_score_)

#using normal method
#======================
test_score = ridge_cv2.score(X_test,y_test)
print(test_score)