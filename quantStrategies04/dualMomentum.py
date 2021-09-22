# dual momentum strategy
# 절대 모멘텀(absolute momentum)
# : 최근 N개월간 수익률이 양수이면 매수하고 음수이면 공매도하는 전략
# 상대 모멘텀(relative momentum)
# : 투자 종목군이 10개 종목이라 할 때 10개 종목의 최근 N개월 모멘텀을 계산해 상대적으로 모멘텀이 높은 종목을 매수하고 상대적으로 낮은 종목은 공매도하는 전략

# 절대 모멘텀 전략
# 여기에서는 과거 12개월간 종가의 수익률을 절대 모멘텀 지수로 계산해 주가 변동이 0% 이상이면 매수 신호가 발생하고 0% 미만이면 매도 신호가 발생하는 코드

import pandas as pd
import numpy as np
import datetime

pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 12)

read_df = pd.read_csv('../contents/SPY.csv')
print(read_df.head(3))

price_df = read_df.loc[:,['Date','Adj Close']].copy()
print(price_df.head(5))

# 월말 데이터 사용 - 월말 종가에 접근
price_df['STD_YM'] = price_df['Date'].map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
print(price_df.head(3))

month_list = price_df['STD_YM'].unique()
month_last_df = pd.DataFrame()
for m in month_list:
    month_last_df = month_last_df.append(price_df.loc[price_df[price_df['STD_YM'] == m].index[-1],:])
# set_index() : index를 Date로 설정
month_last_df.set_index(['Date'],inplace=True)
print(month_last_df.head(4))

# data preprocessing
month_last_df['BF_1M_Adj Close'] = month_last_df.shift(1)['Adj Close']
month_last_df['BF_12M_Adj Close'] = month_last_df.shift(12)['Adj Close']
# fillna() : Nan 값에 0을 넣기
month_last_df.fillna(0, inplace=True)
print(month_last_df.head(15))

# make log(history)
book = month_last_df.copy()
# book.set_index(['Date'],inplace=True)
book['Trade'] = ''
print(book.head(3))

# trading
ticker = 'SPY'
for date in month_last_df.index:
    signal = ''
    # 절대 모멘텀 계산
    momentum_index = month_last_df.loc[date, 'BF_1M_Adj Close'] / month_last_df.loc[date, 'BF_12M_Adj Close'] -1
    # return True or False from absolute momentum : 1M > 12M
    flag = True if ((momentum_index > 0.0) and (momentum_index != np.inf) and (momentum_index != -np.inf)) else False
    if flag:
        # Positive 이면 buy!
        signal = 'buy ' + ticker
    print('date = ',date,' momentum index = ', momentum_index, ' flag = ',flag, ' signal = ',signal)
    book.loc[date, 'Trade'] = signal

print("trade= ",book.tail(40))

def returns(book, ticker):
    # 손익 계산
    rtn = 0.0
    book['return'] = 1
    buy = 0.0
    sell = 0.0
    trade_buy = 'buy ' + ticker
    for date in book.index:
        if book.loc[date, 'Trade'] == trade_buy and book.shift(1).loc[date, 'Trade'] == '':
            # long 진입
            buy = book.loc[date, 'Adj Close']
            print('진입일 = ',date, 'long 진입가격 = ',buy)
        elif book.loc[date, 'Trade'] == trade_buy and book.shift(1).loc[date, 'Trade'] == trade_buy:
            # 보유중
            current = book.loc[date, 'Adj Close']
            rtn = (current - buy) / buy + 1
            book.loc[date, 'return'] = rtn
        elif book.loc[date, 'Trade'] == '' and book.shift(1).loc[date, 'Trade'] == trade_buy:
            # long 청산
            sell = book.loc[date, 'Adj Close']
            # 손익 계산
            rtn = (sell - buy) / buy + 1
            book.loc[date, 'return'] = rtn
            print('청산일 : ',date, ' long 진입가격 : ',buy, ' | long 청산가격 : ', sell, ' | return : ',round(rtn, 4))
        if book.loc[date, 'Trade'] == '':
            buy = 0.0
            sell = 0.0
            current = 0.0
    acc_rtn = 1.0
    for date in book.index:
        if book.loc[date, 'Trade'] == '' and book.shift(1).loc[date, 'Trade'] == trade_buy:
            # long 청산 시
            rtn = book.loc[date, 'return']
            # 누적 수익률 계산
            acc_rtn = acc_rtn * rtn
            book.loc[date:, 'acc return'] = acc_rtn
    print('Accunulated return : ', round(acc_rtn,4))
    return (round(acc_rtn,4))

returns(book,ticker)