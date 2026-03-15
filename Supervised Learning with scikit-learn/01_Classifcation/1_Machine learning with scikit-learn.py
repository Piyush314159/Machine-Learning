import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#file read
df=pd.read_csv("Supervised Learning with scikit-learn/Data_set/telecom_churn_clean.csv")

#plot
plt.figure(figsize=(12, 8))

# Plot points where churn = 0 (red)
plt.scatter(df[df['churn'] == 0]['total_day_charge'], #filtering the data we are taking total_eve_charge where churn==0
            df[df['churn'] == 0]['total_eve_charge'], #simmilarly here
            c='red', label='0', alpha=0.6, s=50)

# Plot points where churn = 1 (blue)
plt.scatter(df[df['churn'] == 1]['total_day_charge'], 
            df[df['churn'] == 1]['total_eve_charge'], 
            c='blue', label='1', alpha=0.6, s=50)

plt.xlabel('total day charge', fontsize=12)
plt.ylabel('total eve charge', fontsize=12)
plt.legend(title='', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

#classifier
X = df[["total_day_charge","total_eve_charge"]].values#it makes an Nx2 array beacuse there is two coloumn elements
y = df["churn"].values
X_new = np.array([[56.8,17.5],[24.4,24.1],[50.1,10.9]])#new values for which we will predict the churn

print(X.shape,y.shape)

#the k-Nearest Neighbour classifier
knn = KNeighborsClassifier(n_neighbors=15)#this checks nearest 15 churn points of every X_new point 
#and returns the majority points(churn=0 or churn=1) as a result
knn.fit(X,y)

#prediction
prediction = knn.predict(X_new)
print(f"Predictions : {prediction}")
