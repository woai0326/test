import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import Lasso


data = pd.read_csv('/Users/ScShen/Desktop/deep_nerual_network/all/train.csv', na_values=['NA', '?'])
test = pd.read_csv('/Users/ScShen/Desktop/deep_nerual_network/all/test.csv', na_values=['NA', '?'])
np.sum(data.isna())
df = data.drop(columns = ['id', 'motor_vol','gear_vol','volume_parts','led_vol'])
finaltest = test.drop(columns = ['id', 'motor_vol','gear_vol','volume_parts','led_vol'])
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

missing_median(df, 'cost')
missing_median(finaltest, 'cost')

shape_ohc = pd.get_dummies(df['shape'])
metal_ohc = pd.get_dummies(df['metal'])
df = pd.concat([df, shape_ohc, metal_ohc],axis = 1)
df = df.drop(columns = ['metal', 'shape'])

test_shape_ohc = pd.get_dummies(finaltest['shape'])
test_metal_ohc = pd.get_dummies(finaltest['metal'])
finaltest = pd.concat([finaltest, test_shape_ohc, test_metal_ohc],axis = 1)
finaltest = finaltest.drop(columns = ['metal', 'shape'])

datause = df.sample(frac = 0.1)

def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    #if target_type in (np.int64, np.int32):
        # Classification
        #dummies = pd.get_dummies(df[target])
        #return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    #else:
        # Regression
    return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

x,y = to_xy(df,'weight')
y = np.reshape(y, y.shape[0])

testx = finaltest.as_matrix().astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#feature selection:lasso
from sklearn import metrics
feature_sele = Lasso(random_state=0, alpha=0.1)
feature_sele.fit(x_train,y_train)
pred_lasso = feature_sele.predict(x_test)
score_lasso = np.sqrt(metrics.mean_squared_error(pred_lasso,y_test))

#svm
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(x_train, y_train)
pred_svr = svr.predict(x_test)
score_svr = np.sqrt(metrics.mean_squared_error(pred_svr,y_test))

#knn
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(weights='distance')
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
score_knn = np.sqrt(metrics.mean_squared_error(pred_knn,y_test))

#tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
pred_dtr = dtr.predict(x_test)
score_dtr = np.sqrt(metrics.mean_squared_error(pred_dtr,y_test))

#randomforest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth= 24, n_estimators= 200,max_features=11)
rfr.fit(x_train, y_train)
pred_rfr = rfr.predict(x_test)
score_rfr = np.sqrt(metrics.mean_squared_error(pred_rfr,y_test))

#grid search
from sklearn.grid_search import GridSearchCV
n_estimators_test = {'n_estimators': np.arange(200,250,10)}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(max_features=11, random_state=10),
                       param_grid = n_estimators_test, scoring='mean_squared_error',cv=5)
gsearch1.fit(x_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

max_depth_test = {'max_depth': np.arange(22,28,2)}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(n_estimators= 220, max_features='sqrt', random_state=10),
                       param_grid = max_depth_test, scoring='mean_squared_error',cv=5)
gsearch2.fit(x_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

max_feature_test = {'max_features': np.arange(9,15,2)}
gsearch3 = GridSearchCV(estimator = RandomForestRegressor(max_depth= 24, n_estimators= 220, random_state=10),
                       param_grid = max_feature_test, scoring='mean_squared_error',cv=5)
gsearch3.fit(x_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

#GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=9, min_samples_split = 30, n_estimators= 300)
gbr.fit(x_train, y_train)
pred_gbr = gbr.predict(x_test)
score_gbr = np.sqrt(metrics.mean_squared_error(pred_gbr,y_test))

max_depth_test = {'max_depth': np.arange(3,14,3)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(),
                       param_grid = max_depth_test, scoring='mean_squared_error',cv=3)
gsearch1.fit(x_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

min_samples_split_test = {'min_samples_split': np.arange(10,100,20)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(max_depth=9),
                       param_grid = min_samples_split_test, scoring='mean_squared_error',cv=3)
gsearch2.fit(x_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

#xgboost
from xgboost import XGBRegressor
xgr = XGBRegressor(max_depth=9,n_estimators=500,min_child_weight=1,gamma=0.3)
xgr.fit(x_train, y_train)
pred_xgr = xgr.predict(x_test)
score_xgr = np.sqrt(metrics.mean_squared_error(pred_xgr,y_test))

max_depth_test = {'max_depth': np.arange(9,20,3)}
gsearch1 = GridSearchCV(estimator = XGBRegressor(),
                       param_grid = max_depth_test, scoring='mean_squared_error',cv=3)
gsearch1.fit(x_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

min_child_weight_test = {'min_child_weight': np.arange(15,20,2)}
gsearch2 = GridSearchCV(estimator = XGBRegressor(max_depth=9,n_estimators=300),
                       param_grid = min_child_weight_test, scoring='mean_squared_error',cv=3)
gsearch2.fit(x_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

gamma_test = {'gamma': np.arange(0.1,0.7,0.2)}
gsearch3 = GridSearchCV(estimator = XGBRegressor(max_depth=9, min_child_weight= 1,n_estimators=300),
                       param_grid = gamma_test, scoring='mean_squared_error',cv=3)
gsearch3.fit(x_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

colsample_bytree_test = {'subsample': np.arange(0.5,1.0,0.1)}
gsearch4 = GridSearchCV(estimator = XGBRegressor(max_depth=9, min_child_weight= 1,n_estimators=300,gamma=0.3),
                       param_grid = colsample_bytree_test, scoring='mean_squared_error',cv=3)
gsearch4.fit(x_train, y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

#nn
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpointer],verbose=0,epochs=1000)
# Predict and measure RMSE
pred_nn = model.predict(x_test)
score_nn = np.sqrt(metrics.mean_squared_error(pred_nn,y_test))


pred = pred_xgr*3/4+pred_rfr/8+pred_gbr/8
score = np.sqrt(metrics.mean_squared_error(pred,y_test))

test_pred_rfr = rfr.predict(testx)
test_pred_xgr = xgr.predict(testx)
test_pred_gbr = gbr.predict(testx)
test_pred = test_pred_xgr*7/16+test_pred_rfr/8+pred_gbr*7/16

online = test['id']
online['weight'] = test_pred
online.to_csv('/Users/ScShen/Desktop/deep_nerual_network/all/submission.csv', index=None)