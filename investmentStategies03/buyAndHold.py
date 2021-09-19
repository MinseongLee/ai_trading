# buy and hold strategy
# 주식을 매수한 후 장기 보유하는 투자 전략

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows',12)
pd.set_option('display.max_columns',12)

df = pd.read_csv('../contents/AAPL.csv', index_col='Date', parse_dates=['Date'])
print(df.head(100))
print("=========================================================================\n")
# 결측(missing value) == NULL
print(df[df.isin([np.nan, np.inf, -np.inf]).any(1)])
print(df.isna().sum())

print("=========================================================================\n")
# 시가보단 보통 전략을 테스트할 때 종가를 기준으로 처리

# data slicing
price_df = df.loc[:,['Adj Close']].copy()
print(price_df)
plt.plot(price_df['Adj Close'])
plt.xlabel('date')
plt.ylabel('adj close')
plt.show()

from_date = '1997-01-02'
to_date = '2021-06-10'
plt.plot(price_df.loc[from_date:to_date])
plt.xlabel('date')
plt.ylabel('adj close')
plt.show()

# 최대 낙폭(maximum draw down)(MDD)
# 최고점 대비 현재까지 하락한 비율 중 최대 하락률

# 일별 수익률
price_df['daily_rtn'] = price_df['Adj Close'].pct_change()
print(price_df.head(5))
# 누적 곱
price_df['st_rtn'] = (1+price_df['daily_rtn']).cumprod()
print(price_df.tail(10))
plt.plot(price_df['st_rtn'])
plt.xlabel('date')
plt.ylabel('st_rtn')
plt.show()

base_date = '2015-01-16'
tmp_df = price_df.loc[base_date:,['st_rtn']] / price_df.loc[base_date,['st_rtn']]
last_date = tmp_df.index[-1]
print(tmp_df.tail(10))
print(last_date)
print(price_df.loc[base_date,['st_rtn']])

print('누적 수익 : ', tmp_df.loc[last_date,'st_rtn'])
plt.plot(tmp_df)
plt.xlabel('date')
# plt.ylabel('st_rtn')
plt.show()