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

'''
상대 모멘텀 전략
relative momentum
'''

import os
import glob
import pandas as pd
import numpy as np
import datetime

# file list를 뽑을 때 사용 glob
files = glob.glob('../contents/*.csv')

# create dataframes
# Monthly
month_last_df = pd.DataFrame(columns=['Date', 'CODE','1M_RET'])
# stocks
stock_df = pd.DataFrame(columns=['Date', 'CODE', 'Adj Close'])

def data_preprocessing(sample, ticker,base_date):
    # add ticker
    sample['CODE'] = ticker
    sample = sample[sample['Date'] >= base_date][['Date','CODE','Adj Close']].copy()
    # 기준일자 이후 데이터 사용
    sample.reset_index(inplace=True, drop=True)
    sample['STD_YM'] = sample['Date'].map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
    # return column
    sample['1M_RET'] = 0.0
    ym_keys = list(sample['STD_YM'].unique())
    return sample, ym_keys

# load each csv file
for csv in files:
    if os.path.isdir(csv):
        print('%s <DIR> '%csv)
    else:
        folder, name = os.path.split(csv)
        head, tail = os.path.splitext(name)
        print('csv = ',csv)
        read_df = pd.read_csv(csv)
        # first, preprocessing
        price_df, ym_keys = data_preprocessing(read_df, head, base_date='2010-01-02')
        # use the data
        stock_df = stock_df.append(price_df.loc[:,['Date', 'CODE','Adj Close']],sort=False)
        # 1개월간 수익률 계산 for 월별 상대 모멘텀 계산
        for ym in ym_keys:
            m_ret = price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1], 'Adj Close'] \
                / price_df.loc[price_df[price_df['STD_YM'] == ym].index[0],'Adj Close']
            price_df.loc[price_df['STD_YM'] == ym, ['1M_RET']] = m_ret
            month_last_df = month_last_df.append(price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],['Date', 'CODE','1M_RET']])

# second, filtering from relative momentum returns
month_ret_df = month_last_df.pivot('Date', 'CODE', '1M_RET').copy()
month_ret_df = month_ret_df.rank(axis=1, ascending=False, method='max', pct=True)
# 상위 40% 안에 드는 종목들만 signal list
month_ret_df = month_ret_df.where(month_ret_df < 0.4, np.nan)
month_ret_df.fillna(0, inplace=True)
month_ret_df[month_ret_df != 0] = 1
stock_codes = list(stock_df['CODE'].unique())

# third, trading + positioning of signal list
def create_trade_book(sample, sample_codes):
    book = pd.DataFrame()
    book = sample[sample_codes].copy()
    book['STD_YM'] = book.index.map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m'))
    # book['trade'] = ''
    # 종목이 다수이므로, 종목에 따라 포지션과 수익률을 기록한다.
    for c in sample_codes:
        book['p '+c] = ''
        book['r '+c] = ''
    return book

sig_dict = dict()
for date in month_ret_df.index:
    ticker_list = list(month_ret_df.loc[date,month_ret_df.loc[date,:] >= 1.0].index)
    sig_dict[date] = ticker_list
stock_c_matrix = stock_df.pivot('Date', 'CODE', 'Adj Close').copy()
book = create_trade_book(stock_c_matrix, list(stock_df['CODE'].unique()))

# positioning
for date, values in sig_dict.items():
    for stock in values:
        book.loc[date, 'p '+stock] = 'ready ' + stock

# print(book.loc['2011-12-28':'2012-03-01',['AAPL', 'p AAPL', 'r AAPL']])

def tradings(book, s_codes):
    std_ym = ''
    buy_phase = False
    for s in s_codes:
        print('s_code= ',s)
        for date in book.index:
            if book.loc[date,'p '+s] == '' and book.shift(1).loc[date,'p '+s] == 'ready ' + s:
                std_ym = book.loc[date, 'STD_YM']
                buy_phase = True
            if book.loc[date, 'p '+s] == '' and book.loc[date, 'STD_YM'] == std_ym and buy_phase == True:
                book.loc[date,'p '+s] = 'buy ' + s
            if book.loc[date,'p '+s] == '':
                std_ym = None
                buy_phase = False
    return book
# exec trading
book = tradings(book, stock_codes)

# print(book.loc['2012-01-27':'2012-03-01',['AAPL', 'p AAPL', 'r AAPL']])

def multi_returns(book, tickers):
    rtn = 1.0
    buy_dict = {}
    num = len(tickers)
    sell_dict = {}

    for date in book.index:
        for ticker in tickers:
            p_status = 'p ' + ticker
            r_status = 'r ' + ticker
            buy_status = 'buy ' + ticker
            ready_status = 'ready ' + ticker
            if book.loc[date, p_status] == buy_status and \
                book.shift(1).loc[date, p_status] == ready_status and \
                book.shift(2).loc[date, p_status] == '':
                # long 진입
                buy_dict[ticker] = book.loc[date, ticker]
            elif book.loc[date, p_status] == '' and book.shift(1).loc[date, p_status] == buy_status:
                # long 청산
                sell_dict[ticker] = book.loc[date, ticker]
                # 손익 계산
                rtn = (sell_dict[ticker] / buy_dict[ticker]) -1
                book.loc[date, r_status] = rtn
                # 수익률 계산
                rtn_percentage = round(rtn * 100,2)
                print('개별 청산일: ',date,'종목 코드 : ',ticker, 'long 진입가격 : ',buy_dict[ticker],'| long 청산가격 : ',sell_dict[ticker], ' | return: ',rtn_percentage,'%')
            # zero position, long 청산
            if book.loc[date, p_status] == '':
                buy_dict[ticker] = 0.0
                sell_dict[ticker] = 0.0

    acc_rtn = 1.0
    for date in book.index:
        rtn = 0.0
        count = 0
        for ticker in tickers:
            p_status = 'p ' + ticker
            r_status = 'r ' + ticker
            buy_status = 'buy ' + ticker
            if book.loc[date,p_status] == '' and book.shift(1).loc[date, p_status] == buy_status:
                # 청산 수익 나오므로,
                count += 1
                rtn += book.loc[date, r_status]
        if (rtn != 0.0) & (count != 0):
            acc_rtn *= (rtn / count) + 1
            # 창산 수익률
            sell_return = round((rtn / count),4)
            print('누적 청산일 : ',date,' 청산 종목수 = ',count,
                  '창산 수익률 = ',sell_return,' 누적 수익률 = ',round(acc_rtn,4))
        # 수익률 계산
        book.loc[date,'acc_rtn'] = acc_rtn
    print('누적 수익률 : ',round(acc_rtn,4))



# returns of tickers
multi_returns(book, stock_codes)
























