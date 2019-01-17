from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

from sklearn.cross_validation import train_test_split
X_train1,X_test1,y_train1, y_test1 = train_test_split(digits.data,digits.target,test_size = 0.25)
from sklearn.preprocessing import StandardScaler
ss1 = StandardScaler()
X_train1 = ss1.fit_transform(X_train1)
X_test1 = ss1.transform(X_test1)

from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train1, y_train1)
lsvc_y_predict = lsvc.predict(X_test1)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

from sklearn.metrics import classification_report
print('lsvc model:', lsvc.score(X_test1, y_test1))
print(classification_report(y_test1, lsvc_y_predict, target_names=digits.target_names.astype(str)))

