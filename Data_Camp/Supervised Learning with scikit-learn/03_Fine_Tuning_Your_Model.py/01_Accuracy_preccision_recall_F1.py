from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#file read
df=pd.read_csv("Supervised Learning with scikit-learn/Data_set/telecom_churn_clean.csv")
X = df.drop("churn", axis=1).values
y = df["churn"].values

knn = KNeighborsClassifier(n_neighbors=7)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=42)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))