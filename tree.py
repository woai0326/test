import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.head(5)
titanic.info()
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
X.info
X['age'].fillna(X['age'].mean(), inplace=True)

from sklearn.cross_validation import train_test_split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.25)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train3 = vec.fit_transform(X_train3.to_dict(orient='record'))
print(vec.feature_names_)
X_test3 = vec.transform(X_test3.to_dict(orient='record'))

#DecisionTree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train3, y_train3)
dtc_y_predict = dtc.predict(X_test3)

from sklearn.metrics import classification_report
print('dtc accuracy:', dtc.score(X_test3, y_test3))
print(classification_report(y_test3, dtc_y_predict, target_names=['died', 'suvived']))

#RandomForest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train3, y_train3)
rfc_y_predict = rfc.predict(X_test3)

print('rfc accuracy:', rfc.score(X_test3, y_test3))
print(classification_report(y_test3, rfc_y_predict, target_names=['died', 'suvived']))

#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train3, y_train3)
gbc_y_predict = gbc.predict(X_test3)

print('dtc accuracy:', gbc.score(X_test3, y_test3))
print(classification_report(y_test3, gbc_y_predict, target_names=['died', 'suvived']))

#xgboost
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train3, y_train3)
print('xgbc accuracy:', xgbc.score(X_test3, y_test3))




