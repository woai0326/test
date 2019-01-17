import pandas as pd
import tensorflow as tf
import base64
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import io
import requests
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    # Regression
    return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

#regression tensorflow
df = pd.read_csv('/Users/ScShen/Downloads/t81_558_deep_learning-master/data/auto-mpg.csv', na_values=['NA', '?'])
df = df.drop(columns='name')
median = df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(median)
x,y = to_xy(df, 'mpg')
model = Sequential()
model.add(Dense(25, input_dim = x.shape[1], activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, verbose=2,epochs=50)

#predict
pred = model.predict(x)
print("Shape: {}".format(pred.shape))
print(pred)
# Measure RMSE error
score = np.sqrt(metrics.mean_squared_error(pred,y))
print(f"Final score (RMSE): {score}")
# Sample predictions
for i in range(10):
    print(f"{i+1}. Car name: {cars[i]}, MPG: {y[i]}, predicted MPG: {pred[i]}")


#classification tensorflow
import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping

def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

url="https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/data/iris.csv"
df=pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')),na_values=['NA','?'])

species = encode_text_index(df,"species")
x,y = to_xy(df,"species")

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(5, activation='relu')) # Hidden 2
model.add(Dense(y.shape[1],activation='softmax')) # Output
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor = 'val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model.fit(x, y, validation_data=(x_test,y_test), callbacks=[monitor], verbose=2, epochs=50)

#predict
pred = model.predict(x)
print("Shape: {pred.shape}")
print(pred[0:5])
predict_classes = np.argmax(pred,axis=1)
print(predict_classes)
expected_classes = np.argmax(y,axis=1)
print(f"Predictions: {predict_classes}")
print(f"Expected: {expected_classes}")

# Accuracy
from sklearn.metrics import accuracy_score
correct = accuracy_score(expected_classes,predict_classes)
print(f"Accuracy: {correct}")