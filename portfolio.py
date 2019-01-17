import tushare as ts
import pandas as pd
import numpy as np
import ffn
from matplotlib import pyplot as plt

hist_data1 = ts.get_hist_data('000001')
hist_data2 = ts.get_hist_data('000002')
hist_data3 = ts.get_hist_data('000005')
hist_data4 = ts.get_hist_data('000004')

def get_profit(df, name):
    df_filled = df['close'].fillna(method='ffill')
    diff = df_filled.diff(-1)
    profit = diff/df_filled.shift(-1)
    profit.name = name
    return profit

profit1 = get_profit(hist_data1, '000001')
profit2 = get_profit(hist_data2, '000002')
profit3 = get_profit(hist_data3, '000005')
profit4 = get_profit(hist_data4, '000004')
input_data = pd.concat([profit1, profit2, profit3, profit4], axis=1)
input_data = input_data[input_data.index>'2016-12-31']
input_data = input_data.dropna(axis=0, how='any',)

cr = (1+input_data).cumprod()
cr.plot()
input_data.plot()
input_data.corr()

from scipy import linalg
class MeanVariance:
    def __init__(self,returns):
        self.returns = returns

    def minVar(self,goalRet):
        covs = np.array(self.returns.cov())
        means = np.array(self.returns.mean())
        L1 = np.append(np.append(covs.swapaxes(0,1),[means],0),[np.ones(len(means))],0).swapaxes(0,1)
        L2 = list(np.ones(len(means)))
        L2.extend([0,0])
        L3 = list(means)
        L3.extend([0, 0])
        L4 = np.array([L2,L3])
        L = np.append(L1,L4,0)
        results = linalg.solve(L,np.append(np.zeros(len(means)),[1,goalRet],0))
        return(np.array([list(self.returns.columns),results[:-2]]))

    def frontierCurve(self):
        goals = [x/200000 for x in range (-100,4000)]
        variances = list(map(lambda x:self.calVar(self.minVar(x)[1,:].astype(np.float)),goals))
        plt.plot(variances,goals)

    def meanRet(self,fracs):
        meanRisky = ffn.to_returns(self.returns).mean()
        assert len(meanRisky) == len(fracs), 'Length of fractions must be equal to number of assets'
        return(np.sum(np.multiply(meanRisky, np.array(fracs))))

    def calVar(self,fracs):
        return(np.dot(np.dot(fracs,self.returns.cov()),fracs))


minVar = MeanVariance(input_data)
minVar.frontierCurve()

input_data.set_index(pd.to_datetime(input_data.index), drop=True, inplace=True)


train_set = input_data['2017']
test_set = input_data['2018']

varMinimizer = MeanVariance(train_set)
goal_return = 0.003
portfolio_weight=varMinimizer.minVar(goal_return)


test_return = np.dot(test_set,np.array([portfolio_weight[1,:].astype(np.float)]).swapaxes(0,1))
test_return = pd.DataFrame(test_return,index=test_set.index)
test_cum_return = (1+test_return).cumprod()









