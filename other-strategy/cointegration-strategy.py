# from chapter 4 algorithmic trading by Ernest P. Chan

from ib_insync import *
import pandas as pd
import os

util.startLoop()
ib=IB()
ib.connect('127.0.0.1',7497,clientId=12)
data_dir="Data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# strategy on SPY and QQQ, two highly correlative ETFs

# get historical data for SPY and QQQ, using day interval
spy=Stock('SPY', 'SMART', 'USD')
qqq=Stock('QQQ', 'SMART', 'USD')

bars_spy=ib.reqHistoricalData(spy,endDateTime="",barSizeSetting="1 day",durationStr="2 Y" ,whatToShow="TRADES",useRTH=False,formatDate=1)

bars_qqq=ib.reqHistoricalData(qqq,endDateTime="",barSizeSetting="1 day",durationStr='2 Y',whatToShow="TRADES",useRTH=False,formatDate=1)

df_spy=util.df(bars_spy)[['date','close','volume','high','low']]
df_qqq=util.df(bars_qqq)[['date','close','volume','high','low']]
df_spy.to_csv(data_dir+"spy.csv",index=False)
df_qqq.to_csv(data_dir+"qqq.csv",index=False)



# statioanry test (ADF)

from statsmodels.tsa.stattools import adfuller

adf_spy=adfuller(df_spy['close'])
adf_qqq=adfuller(df_qqq['close'])

print(f"SPY ADF p-value: {adf_spy[1]}")
print(f"QQQ ADF p-value: {adf_qqq[1]}")
"""
SPY ADF p-value: 0.4891571692373315
QQQQ ADF p-value: 0.3165152796366484
"""
# cointegration test

from statsmodels.tsa.stattools import coint
score,pvalue,_=coint(df_spy['close'],df_qqq['close'])
print(f"Cointegration score: {score}")
print(f"Cointegration pvalue: {pvalue}")
"""
Cointegration score: -3.6773100983876903
Cointegration pvalue: 0.019592819623051174
"""

# signal generation by spread
look_back=20
import statsmodels.api as sm
#get hedge ratio
X=sm.add_constant(df_qqq['close'])
y=df_spy['close']
model=sm.OLS(y,X).fit()
beta=model.params['close']

# calculate spread
df = pd.merge(df_spy[['date', 'close']], df_qqq[['date', 'close']], on='date', suffixes=('_SPY', '_QQQ'))
df.set_index('date', inplace=True)

df['spread'] = df['close_SPY'] - beta * df['close_QQQ']


# z-score spread
df['spread_mean']=df['spread'].rolling(look_back).mean()
df['spread_std']=df['spread'].rolling(look_back).std()
df['z_score']=(df['spread']-df['spread_mean'])/df['spread_std']

df['long_entry']=df['z_score']<-1
df['short_entry']=df['z_score']>1
df['exit']=df['z_score'].abs()<0.1
df.to_csv(data_dir+"df.csv")
#backtest
position=0
pnl=[]
entry_price=0
for i in range(len(df)):
    if position==0:
        if df['long_entry'].iloc[i]:
            position=100
            entry_price=df['spread'].iloc[i]
        elif df['short_entry'].iloc[i]:
            position=-100
            entry_price=df['spread'].iloc[i]
        pnl.append(0)

    elif df['exit'].iloc[i]:
        pnl.append((df['spread'].iloc[i]-entry_price)*position)
        position=0
        entry_price=0
    else:
        pnl.append(0)

df['pnl']=pnl
df['cum_pnl']=df['pnl'].cumsum()


#plot result
import matplotlib.pyplot as plt
df['cum_pnl'].plot(title="Cointegration",figsize=(12,6))
plt.ylabel('Cum PNL')
plt.grid()
plt.show()







# bonus: mean-reversion: # Chan's strategy: if (today's open - yesterday's low)/yesterday's low < -1 std, and current open is above 20 day ma,
# # then buy, all position close by market close