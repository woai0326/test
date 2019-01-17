from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])
...


from sklearn.datasets import load_iris
iris = load_iris()
iris.data.shape

from sklearn.cross_validation import train_test_split
X_train2,X_test2,y_train2,y_test2 = train_test_split(iris.data,iris.target,test_size=0.25)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train2 = ss.fit_transform(X_train2)
X_test2 = ss.fit_transform(X_test2)

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=6)
knc.fit(X_train2,y_train2)
knc_y_predict = knc.predict(X_test2)

print('knc accu:',knc.score(X_test2,y_test2))
from sklearn.metrics import classification_report
print(classification_report(y_test2,knc_y_predict,target_names=iris.target_names))
