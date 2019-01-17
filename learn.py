import numpy as np
import sklearn
from sklearn import datasets
from sklearn .cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import OneHotEncoder



ssc = pd.read_excel('/Users/ScShen/Desktop/1.xlsx')
# X = ssc.iloc[:,:2]
# y = ssc.iloc[:,-1]


pd.get_dummies(ssc)

iris = datasets.load_iris()
# datasets.load_boston()


iris_X = iris.data
iris_y = iris.target



print(iris_X[:2, :])
print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

print(y_train) # 分开的同时也被打乱了


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)


print(sklearn.metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))
