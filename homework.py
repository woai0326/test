import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers.core import Dense, Activation

data = pd.read_csv('/Users/ScShen/Downloads/t81_558_deep_learning-master/data/reg-30-spring-2018.csv')
data = data.drop(columns='id')
med = data['width'].median()
data['width'] = data['width'].fillna(med)
data['landings']=zscore(data['landings'])
data['number']=zscore(data['number'])
data['pack']=zscore(data['pack'])

data['density'] = data['weight']/data['volume']

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

encode_text_dummy(data,'region')

def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

encode_text_index(data, 'item')

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


df = data.drop(columns=['id','target'])
for i in range(len(df.dtypes)):
    if df.dtypes[i] == object:
        encode_text_dummy(df, df.columns[i])
    else:
        df[df.columns[i]] = zscore(df[df.columns[i]])
df['weight'] = zscore(df['weight'])
df['volume'] = zscore(df['volume'])
df['width'] = data['width']
med = data['width'].median()
df['width'] = df['width'].fillna(med)
df['width'] = zscore(df['width'])
df = pd.concat([df,data['target']],axis=1)

x,y = to_xy(df, 'target')

model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
model.add(Dense(25, input_dim = x.shape[1], activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, verbose=2,epochs=250)

pred = model.predict(x)
print("Shape: {}".format(pred.shape))
print(pred)
df['pred'] = pred
predict = df['pred']
ans = pd.concat([predict,data['id']],axis=1)


from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
model = Sequential()
model.add(Dense(20, input_dim = x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer= 'adam')
monitor = EarlyStopping(monitor='Val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor, checkpointer], verbose=2,epochs=1000)
model.load_weights('best_weights.hdf5') # load weights from best model

# Predict and measure RMSE
from sklearn import metrics
pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Score (RMSE): {}".format(score))

# Plot the chart
import matplotlib.pyplot as plt
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

chart_regression(pred.flatten(),y_test, sort=False)

#crossvalidation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
result = pd.DataFrame(columns=['test_index','pred'])
temp_result = pd.DataFrame(columns=['test_index','pred'])
score = 0
i = 0
for train_index, test_index in kf.split(df):
    #print("TRAIN:", train_index, "TEST:",test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = Sequential()
    model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)  # save best model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor, checkpointer], verbose=0,
              epochs=300)
    model.load_weights('best_weights.hdf5')

    pred = model.predict(x_test)
    test_index = np.transpose([test_index]).astype(int)
    temp_result=pd.DataFrame(np.concatenate((test_index, pred),axis=1), columns=['test_index','pred'])
    #temp_result['test_index'] = test_index
    #temp_result['pred'] = pred  ???
    result = pd.concat([result, temp_result],axis = 0)

    temp_score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    score = score + temp_score
    i=i+1
    print(i)