from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
#logistic regression
#===========================

df=pd.read_csv("Supervised Learning with scikit-learn/Data_set/telecom_churn_clean.csv")
X = df.drop("churn", axis=1).values
y = df["churn"].values

logreg = LogisticRegression(max_iter=50000)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=42)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print(y_pred_prob[0])

#ROC Curve
#================

fpr , tpr, thresholds = roc_curve(y_test,y_pred_prob) #fpr-first positive rate , tpr-true positive rate

plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr)
plt.xlabel("First Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()

print(roc_auc_score(y_test,y_pred_prob))