from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
iris = datasets.load_iris()


iris = pd.DataFrame(
    data = np.c_[iris['data'], iris['target']],
    columns = iris['feature_names'] + ['target']
)

species = []
for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append('setosa')
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')
        
iris['species'] = species


X = iris.drop(['target','species'], axis=1)
X = X.to_numpy()[:, (2,3)]
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)
# train the model
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

training_prediction = log_reg.predict(X_train)
test_prediction = log_reg.predict(X_test)


print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recall scores
print(metrics.classification_report(y_train, training_prediction, digits=3))
# Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))

print("Precision, Recall, Confusion matrix, in testing\n")

# Precision Recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))
# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))
