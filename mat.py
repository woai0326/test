import tushare as ts
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from arch.unitroot import ADF
from statsmodels.tsa import arima_model
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import *

index = ts.get_hist_data('sh')
shindex = index['close']
shindex = shindex.diff(-1)/shindex.shift(-1)

shindex.index = pd.to_datetime(shindex.index)

shindex = shindex.dropna()

acf = stattools.acf(shindex)
pacf = stattools.pacf(shindex)
plot_acf(shindex,use_vlines=True,lags=30)
plot_pacf(shindex,use_vlines=True,lags=30)

shindex.plot()

adfshindex = ADF(shindex)
print(adfshindex.summary().as_text())

whitenoise = np.random.standard_normal(500)
plt.plot(whitenoise,c = 'b')

cpi = ts.get_cpi()
cpi.index = pd.to_datetime(cpi['month'])
cpi = cpi['cpi']
cpitrain = cpi['2016-01-01':'2000-01-01']
cpitrain.plot()
#是否平稳
print(ADF(cpitrain,max_lags=10).summary().as_text())
#是否白噪声
ljb0 = stattools.q_stat(stattools.acf(cpitrain)[1:12],len(cpitrain))
ljb0[1][-1]

#识别ARMA模型参数pq
plot_acf(cpitrain,use_vlineEs=True,lags=30)
plot_pacf(cpitrain,use_vlines=True,lags=30)

model1 = arima_model.ARIMA(cpitrain.values,order=(1, 0, 1)).fit()
model1.summary()

p = np.arange(1,4)
q = np.arange(1,4)
result = dict()
for i in p:
    for j in q:
        model1 = arima_model.ARIMA(cpitrain.values, order=(i, 0, j)).fit()
        result[(i,j)] = (model1.aic, model1.bic)

bestmodel = arima_model.ARIMA(cpitrain.values,order=(3, 0, 2)).fit()
stdresid = bestmodel.resid/math.sqrt(bestmodel.sigma2)
plt.plot(stdresid)
plot_acf(stdresid,lags=12)
ljb = stattools.q_stat(stattools.acf(stdresid)[1:12],len(stdresid))
ljb[1][-1]

bestmodel.forecast(3)[0]
a = cpi['2016-05-01':'2016-02-01']

