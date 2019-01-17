from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import shutil
import os
import requests
import base64

#help function
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = "{}-{}".format(name, tv)
        df[name2] = l


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low

#read data
data = pd.read_csv('/Users/ScShen/Downloads/t81_558_deep_learning-master/data/cls-32-spring-2018.csv')
np.sum(data.isna())
missing_median(data, 'weight')
data.drop('id',1,inplace=True)
types = data.dtypes
columns = data.columns
for i in range(len(types)-1):
    if types[i] == np.dtype('O'):
        # print(data.dtypes[i])
        encode_text_dummy(data, columns[i])
    else:
        # pass
        data[columns[i]] = zscore(data[columns[i]])


data['height']=zscore(data['height'])
data['distance']=zscore(data['distance'])
data['usage']=zscore(data['usage'])
data['dose']=zscore(data['dose'])
data['count']=zscore(data['count'])
data['max']=zscore(data['max'])
data['weight']=zscore(data['weight'])
data['age']=zscore(data['age'])
encode_text_dummy(data,'country')
encode_text_dummy(data,'dock')
encode_text_dummy(data,'rank')
encode_text_dummy(data,'function')
encode_text_dummy(data,'cat1')
encode_text_dummy(data, 'class')
encode_text_index(data, 'target')

use = data[:500]

x,y = to_xy(use,"target")

#cross-validation
kf = KFold(5)

oos_y = []
oos_pred = []
fold = 0

for train, test in kf.split(x):
    fold += 1
    print("Fold #{}".format(fold))

    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1], activation='relu'))  # Hidden 1
    model.add(Dense(25, activation='relu'))  # Hidden 2
    model.add(Dense(y.shape[1], activation='softmax'))  # Output
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=25, verbose=1, mode='auto')

    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=0, epochs=1000)

    pred = model.predict(x_test)

    oos_y.append(y_test)
    pred = np.argmax(pred, axis=1)  # raw probabilities to chosen class (highest probability)
    oos_pred.append(pred)

    # Measure this fold's accuracy
    y_compare = np.argmax(y_test, axis=1)  # For accuracy calculation
    score = metrics.accuracy_score(y_compare, pred)
    print("Fold score (accuracy): {}".format(score))

# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
oos_y_compare = np.argmax(oos_y, axis=1)  # For accuracy calculation

score = metrics.accuracy_score(oos_y_compare, oos_pred)
print("Final score (accuracy): {}".format(score))

predict = oos_pred.transpose()
use['pred'] = predict
predict = use['pred']
result = pd.concat([predict,use['target']],axis = 1)