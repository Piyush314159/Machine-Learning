from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#file read
df=pd.read_csv("Supervised Learning with scikit-learn/Data_set/telecom_churn_clean.csv")


X = df[["total_day_charge","total_eve_charge"]].values # Feature matrix: shape (N, 2) — converted to NumPy array via .values
y = df["churn"].values

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21, stratify=y) 

train_accuracies = {}
test_accuracies = {}
neighbors=np.arange(1,30)
for neighbor in neighbors:
    knn= KNeighborsClassifier(n_neighbors = neighbor)
    knn.fit(X_train,y_train)
    train_accuracies[neighbor] = knn.score(X_train,y_train)
    test_accuracies[neighbor] = knn.score(X_test,y_test)

#plotting
plt.figure(figsize=(8,6))
plt.title("KNN : Varyig Number of Neighbors")
plt.plot(neighbors,train_accuracies.values(), label="Training Accuracies")
plt.plot(neighbors,test_accuracies.values(), label = "Testing Accuracies")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()