# 평균 회귀 전략(regression to mean)
# 주가의 평균가격을 계산해 이 값을 기준으로 매매를 결정

# 볼린저 밴드(bollinger band)
# 현재 주가가 상대적으로 높은지 낮은지를 판단할 때 사용하는 보조지표
# 3개 선으로 이루어짐
# 상단 밴드 = n일 이동 평균선 + 2 * n일 이동 표준 편차
# 중간 밴드 = n일 이동 평균선
# 하단 밴드 = n일 이동 평균선 + 2 * n일 이동표준 편차
# 특징
# : 추세와 변동성을 분석해주고 평균 회귀 현상을 관찰하는데도 유용

# formula
# https://www1.oanda.com/forex-trading/learn/trading-tools-strategies/bollinger

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 12)
df = pd.read_csv('../contents/AAPL.csv')
print(df.describe())
print("=========================================================================\n")
price_df = df.loc[:,['Date','Adj Close']].copy()
print(price_df.head(3))
print("=========================================================================\n")
# index settings
price_df.set_index(['Date'],inplace=True)
print(price_df.head(2))
print("=========================================================================\n")

# make bollinger band

# 결측치 처리는 실무에서는 쿠션 데이터(cushion data)라고 불리는 여분 데이터를 확보해서 결측치를 채워넣어줘야한다.
def bollinger_band(data, n, sigma):
    bol = data.copy()
    # 중간 밴드 = n일 이동평균선
    bol['mb'] = bol['Adj Close'].rolling(window= n).mean()
    # up band
    bol['ub'] = bol['mb'] + sigma * bol['Adj Close'].rolling(window= n).std()
    # down band
    bol['db'] = bol['mb'] - sigma * bol['Adj Close'].rolling(window= n).std()
    return bol

bollinger = bollinger_band(price_df, n=20,sigma=2)

base_date = '2009-01-02'
sample = bollinger.loc[base_date:]
print(sample.head(5))
print("=========================================================================\n")

# 진입, 청산 신호가 발생할 때, 기록하는 거래 장부(trading book)
def trading_book(data):
    book = data[['Adj Close']].copy()
    book['trade'] = ''
    return (book)

book = trading_book(sample)

# buy and sell : 평균가격 기준
def tradings(data, book):
    for i in data.index:
        # 상단 밴드 이탈 시 동작 안함.
        if data.loc[i, 'Adj Close'] > data.loc[i, 'ub']:
            book.loc[i, 'trade'] = 'sell'
        # 하단 밴드 이탈 시
        elif data.loc[i, 'db'] > data.loc[i, 'Adj Close']:
            book.loc[i, 'trade'] = 'buy'
        # 횡보 - 볼린저 밴드 내부
        elif data.loc[i, 'ub'] >= data.loc[i, 'Adj Close'] and data.loc[i, 'Adj Close'] >= data.loc[i, 'db']:
            # if bought before
            if book.shift(1).loc[i, 'trade'] == 'buy':
                # stay "buy status"
                book.loc[i, 'trade'] = 'buy'
            else:
                book.loc[i, 'trade'] = 'sell'
    return (book)

book = tradings(sample, book)
print(book.tail(100))
print("=========================================================================\n")

# 전략 수익률
# 전체 수익률(rate of return) 계산 func
def returns(book):
    # 손익 계산
    rtn = 1.0
    book['return'] = 1
    buy_price = 0.0
    sell_price = 0.0
    for date in book.index:
        # long 진입
        if book.loc[date, 'trade'] == 'buy' and book.shift(1).loc[date, 'trade'] == 'sell':
            buy_price = book.loc[date, 'Adj Close']
            print('진입 : ',date, ' long 진입가격 : ', buy_price)
        # long 청산
        elif book.loc[date, 'trade'] == 'sell' and book.shift(1).loc[date, 'trade'] == 'buy':
            sell_price = book.loc[date, 'Adj Close']
            # 손익 계산
            rtn = (sell_price - buy_price) / buy_price + 1
            book.loc[date, 'return'] = rtn
            print('청산일 : ',date, ' long 진입가격 : ',buy_price,' long 청산가격 : ',sell_price,' return: ',round(rtn,4))
        if book.loc[date, 'trade'] == 'sell':
            buy_price = 0.0
            sell_price = 0.0
    acc_rtn = 1.0
    for date in book.index:
        rtn = book.loc[date, 'return']
        # 누적 수익률 계산
        acc_rtn = acc_rtn * rtn
        book.loc[date, 'acc return'] = acc_rtn

    print('Accunulated return : ',round(acc_rtn,4))
    return (round(acc_rtn,4))
# 누적수익률이 2.95 : 즉, 초기 투자금의 약 3배의 수익을 본다.
print(returns(book))
print("=========================================================================\n")

plt.plot(book['acc return'])
plt.xlabel('date')
plt.ylabel('acc returns')
plt.show()












