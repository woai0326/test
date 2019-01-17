import tushare as ts
import pandas as pd
import numpy as np

zgtb = ts.get_hist_data('601601')
bldc = ts.get_hist_data('600048')
zgrs = ts.get_hist_data('601628')

zgtb = zgtb['close']
bldc = bldc['close']
zgrs = zgrs['close']
zgtb.index = pd.to_datetime(zgtb.index)
bldc.index = pd.to_datetime(bldc.index)
zgrs.index = pd.to_datetime(zgrs.index)
zgtb = zgtb['2018-09-07':'2018-01-01']
bldc = bldc['2018-09-07':'2018-01-01']
zgrs = zgrs['2018-09-07':'2018-01-01']
pairf = pd.concat([bldc, zgtb, zgrs], axis = 1)
pairf.columns = ['bldc', 'zgtb', 'zgrs']

def SSD(priceX,priceY):
    returnX = (priceX - priceX.shift(1))/priceX.shift(1)[1:]
    returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
    standardX = (returnX+1).cumprod()
    standardY = (returnY + 1).cumprod()
    SSD = np.sum((standardX-standardY)**2)
    return (SSD)

dis = SSD(bldc,zgtb)


pool = [zgtb, zgrs, bldc]
result = dict()
for i in range(3):
    for j in range(i+1, 3):
        a = pool[i]
        b = pool[j]
        dis = SSD(a, b)
        result[i,j] = dis

