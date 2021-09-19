# 금융 데이터 분석
# datetime, numpy, pandas 에서 다루는 시간

# datetime 핵심 객체
# time, date, datetime, timedelta

# python 내장 객체 datetime
# datetime : 마이크로초 (10^-6)
import datetime
format = '%Y-%m-%d %H:%M:%S'
datetime_str = '2020-07-20 12:40:37'
# strptime() string => datetime type
datetime_dt = datetime.datetime.strptime(datetime_str, format)
print(type(datetime_dt))
print(datetime_dt)

# strftime() datetime => string type
datetime_str = datetime_dt.strftime(format)
print(type(datetime_str))
print(datetime_str)

# numpy library
# datetime64 : 아토초(10^-18)
# date, time 을 하나의 객체로 관리
# pandas는 numpy library 기반으로 만들어짐.

# numpy 날짜 생성 방식 두 가지
# 1) ISO 8601 국제 표준 방식으로 str 타입 값 전달
# 2) unix 시간 이용 - 1970.01.01 부터 시간을 초 단위로 환산


import numpy as np
# https://ko.wikipedia.org/wiki/ISO_8601
# ISO 8601
print(type(np.datetime64('2021-03-28')))
print(np.datetime64('2021-03-28'))
print("-----------------------------------------")

# unix
print(type(np.datetime64(1000, 'ns')))
# ns : 나노초, D : 일, s : 초 단위
print(np.datetime64(1000, 'ns'))
print(np.datetime64(10000, 'D'))
print(np.datetime64(100000000, 's'))
print("-----------------------------------------")

#array()
print(np.array(['2021-03-28','2020-02-28','2015-10-10'], dtype='datetime64'))
print("-----------------------------------------")
# arange() 날짜 범위 지정 D = day, M = month, Y = year
print(np.arange('2020-02', '2020-03', dtype='datetime64[D]'))
print("-----------------------------------------")
print(np.arange('2015-02', '2020-03', dtype='datetime64[M]'))
print("-----------------------------------------")
print(np.arange('2015-02', '2020-03', dtype='datetime64[Y]'))
print("-----------------------------------------")

# 두 날짜 간 간격 : 내부적으로 timedelta 가 실행되므로 산술연산으로 날짜를 구할 수 있다.
print(np.datetime64('2015-01-01') - np.datetime64('2014-02-10'))
print("-----------------------------------------")
print(np.datetime64('2015') - np.datetime64('2014-02'))
print("-----------------------------------------")
print(np.datetime64('2020') - np.datetime64('2014'))

# pands library


import pandas as pd
# date times of numpy
print(pd.Timestamp(1239.1253, unit='D'))
print(pd.Timestamp('2013-01-23'))
print(pd.to_datetime('2013-01-23 09'))
print(pd.to_datetime(['2020-05-10', '2020-06-20']))
print(pd.date_range('2020-05-10','2020-06-20'))

print("---------------------------------------------------------------------------------------------------")

# Time spans
print(pd.Period('2019-01'))
print(pd.Period('2018-10', freq='D'))
print(pd.period_range('2021-02', '2021-03', freq='D'))

# date time 은 한 시점이고, Time spans는 시작 시점부터 종료 시점까지를 나타낸다.
p = pd.Period('2021-06-20')
today = pd.to_datetime('2021-06-20 10')
print(p.start_time <= today <= p.end_time)

print("---------------------------------------------------------------------------------------------------")

# 다양한 주기 표현
# freq=''
# None : 달력상 1일 간격
# B : 영업일(평일), Business Day
# W : weekends = Sunday
# M : the last day of each month
# MS : the first day of each month
# BM : 각 달의 평일 중에서 마지막 날
# BMS : 각 달의 평일 중에서 첫 날
# W-MON : 주-(월요일)
print(pd.date_range('1990-01','1990-02', freq='B'))
print(pd.date_range('2021-09','2021-10', freq='W'))
print(pd.date_range('2021-09','2021-10', freq='W-MON'))

# pandas 데이터 전처리 과정 필수 내용
# 판다스는 데이터 전처리 및 시계열 분석을 효율적으로 수행하기 위해 만든 라이브러리

# reading and writing
# fit data type like csv,, etc
# read : read_csv 
# write : to_csv
import pandas as pd
# https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
df = pd.read_csv('../contents/AAPL.csv')
# limit 5
print(df.head())

# Series : 1차원 배열
# DataFrame : 2차원 배열
print("------------------------------------------------------------------------")

# index_col : index로 Date로 설정, parse_dates : Date의 타입을 Timestamp 로 변경
aapl_df = pd.read_csv('../contents/AAPL.csv', index_col='Date', parse_dates=['Date'])
print(aapl_df.head())
print(type(aapl_df.index))
print(type(aapl_df.index[0]))

# 에러의 주된 원인 제거 - missing value와 이상치 다루기 (preprocessing)
# 데이터 수집 과정에서 사람의 실수나 전산 오류 등 여러 가지 이유로 결측치(missing value)가 발생하는 일이 많다.
# missing value 확인하는 방법과 처리하는 방법

# missing value 확인하는 방법
# NaN, 무한값(infinity) == missing value
s1 = pd.Series([1,np.nan,3,4,5])
s2 = pd.Series([1,2,np.nan,4,5])
s3 = pd.Series([1,2,3,np.nan,5])
df = pd.DataFrame({
    'S1':s1,
    'S2':s2,
    'S3':s3
})
print(df.head())

print("------------------------------------------------------------------------")
# nan값을 확인하는 함수는 isna(), isnull(), isin()
print(df['S1'].isnull())

print(df.isna())
# True = 1 , False = 0
print(df.isnull().sum())
print("------------------------------------------------------------------------")
# isin() : 각 컬럼(Series) 값을 리스트로 parameter로 전달받음 그래서 이 값을 바탕으로 bool 값을 리턴
print(df.isin([np.nan]))

print(df.isin([np.nan]).sum())

# missing value를 처리하는 방법
# 1. 다른 값으로 채우는 방법, 2. 제거하는 방법
# replace nan(null) to sth
print(df.fillna(0))
print("------------------------------------------------------------------------")
# pad / ffill : 앞선 행의 값으로 대체
# bfill / backfill : 다음 행의 값으로 대체
print(df.fillna(method='pad'))
print("------------------------------------------------------------------------")
print(df.fillna(method='bfill'))

# backword fill 방법은 사전 관찰 편향(look-ahead-bias) 문제 발생 가능.
# 그래서 forward fill 방법을 주로 사용
# 혹은 경우에 따라 backword 와 forward 평균 값을 사용할 수도 있다.
print("------------------------------------------------------------------------")
# 삭제(지양하는 방법)
# drop
print(df.dropna())
# axis=0 : row, axis=1 : column
print(df.dropna(axis='rows'))
print(df.dropna(axis=1))

# nan, 무한값을 모두 가져오는 코드.
# isin([]) : 전달받은 리스트 값이 데이터 프레임에 속해있는지 확인
print("결측치?------------------------------------------------------------------------")
print(aapl_df[aapl_df.isin([np.nan, np.inf, -np.inf]).any(1)])

# 데이터 선택 방법 - slicing, indexing, 서브셋 data 추출
# slicing : indexing(위치정보)를 기반으로 데이터를 잘라냄.
# 원하는 데이터 추출할 때 slicing과 indexing을 많이 사용

# 서브셋 데이터
arr = [0,1,2,3,4,5,6,7,8,9]
# 이렇게 원본 데이터에서 임의의 크기를 짤라내어 다른 객체에 저장을 해 놓은 데이터를 서브셋 데이터라고 함.
subset = arr[0:5]
print(subset)

import pandas as pd
df = pd.read_csv('../contents/AAPL.csv', index_col='Date', parse_dates=['Date'])
print(df['Open'].head())
print("------------------------------------------------------------------------")
print(df[['Open', 'High']].head())
print("------------------------------------------------------------------------")
# row 단위 : 주가 분석 데이터에서 행은 날짜를 나타내는 경우가 많다.
print(df[0:3])
print("------------------------------------------------------------------------")

# df 변수의 인덱스 타입은 DatetimeIndex이고, 인덱스를 이루는 개별 요소들의 타입은 Timestamp
# 두 타입이 다른데도 비교연산을 할 수 있는 이유 : loc, iloc를 활용한 인덱싱 방식
# 실무에서 판다스의 loc와 iloc 인덱서를 이용해 데이터 추출 작업을 많이함.
# loc : 인덱스 값을 기준으로 행 데이터를 추출
# iloc : 정수형 숫자로 데이터를 추출
print(df['2021-09-01':'2021-09-10'])

import pandas as pd
df = pd.read_csv('../contents/AAPL.csv')
df.head()
print(type(df.index))
print(type(df.index[0]))
print("------------------------------------------------------------------------")

df = pd.read_csv('../contents/AAPL.csv', index_col='Date', parse_dates=['Date'])
df.head()
print(type(df.index))
print(type(df.index[0]))
print("------------------------------------------------------------------------")

#  두 개의 값이 정확히 같다.
print(df.loc['2020-09-16'])
print(df.iloc[0])
print("------------------------------------------------------------------------")
print(type(df.loc['2020-09-16']))

# loc, iloc 사용방법
print(df.loc['2021-09-01':'2021-09-10',['Open','High','Low','Close']])
print("------------------------------------------------------------------------")
print(df.iloc[100:110,[0,1,2,3]])

# Date를 인덱스로 사용하는 주가 데이터 분석에서는 iloc보다 loc를 사용해 명확한 기간을 추출할 것을 권장
# 왜냐하면 iloc는 데이터 수가 많을 때, 원하는 데이터를 정확하게 추출하기 어려우므로

# str 타입과 Timestamp 타입이 다름에도 비교연산이 될 수 있는 이유
# ISO 8601 표준 방식을 따른다면 내부적으로 데이터 타입이 변환되어 날짜 비교연산이 가능해진다.

# 금융 시계열 데이터 분석에 유용한 판다스 함수
pd.set_option('display.max_rows',12)
pd.set_option('display.max_columns',12)
# shift() func***
# 인덱스에 연결된 데이터를 일정 간격으로 이동시키는 함수
# shift() 로 N일 전 데이터를 가져옴 defulat n(period) = 1
# shift on the bottom
aapl_df['Close_lag1'] = aapl_df['Close'].shift(periods=2)
# shift on the up
aapl_df['Close_lag2'] = aapl_df['Close'].shift(periods=-1)
print(aapl_df.head())

# pct_change() func : 변화율 계산***
# 현재 값과 이전 요소 값의 백분율 변화량을 연산하는 함수
# 주가 데이터를 다룰 때, 수익률을 쉽게 계산 가능.

# period : 숫자 간격만큼 데이터와 백분율 계산 
# 양의 정수 = down, 음의 정수 = up
aapl_df['pct_change'] = aapl_df['Close'].pct_change(periods=1)
aapl_df.head(10)

# diff() func : 변화량 계산***
# 현재 값에서 이전 값을 차감하는 형식으로 변화량을 구함.
aapl_df['Close_diff'] = aapl_df['Close'].diff(periods=1)
aapl_df.head()

# rolling() 함수
# 이평션을 계산
# window 파라미터는 일정 구간 데이터들의 평균값, 최소값, 최대값 등을 계산하는 함수
# 대표적으로 이동 평균선, 지수 이동 평균, 볼린저 밴드를 계산할 때 응용 가능.

# https://pandas.pydata.org/pandas-docs/stable/reference/window.html
# Moving Average
# 크기가 5인 윈도우를 한 칸씩 오른쪽으로 rolling하도록 설정. 그래서 구간 평균, 최댓값, 최소값 등을 구할 수 있음.
df['MA'] = df['Close'].rolling(window = 5).mean()
print(df.head(10))
print("------------------------------------------------------------------------")
print(type(df['Close'].rolling(window=5)))

# resample() func
# 시간 간격을 재조정하는 기능을 함.

# 주가 데이터를 분석하다 보면 데이터는 가능한 한 가장 작은 단위로 수집하는 경향이 있음.
# 예를 들어, 예측 추기를 일(daily)에서 주(weekly)로 바꾼다거나 아니면 월(monthly)이나 연(yearly)단위로 어떤 축약한 내용을 확인하고 싶거나
# 또한 시가(Open), 고가(High), 저가(Low), 종가(Close) (OHLC) 가격의 평균 등
# groupby()보다 resample() of pandas 가 편하고 정말 실무에서 자주 쓰임.

# up-sampling : 분 단위, 초 단위로 샘플의 빈도수를 증가시킨다.
# down-sampling : 몇 일, 몇 달 단위로 샘플의 빈도수를 감소시킨다.
# up-sampling은 보간법(interpolation)을 사용해 누락된 데이터를 채워나감.
# down-sampling은 기존 데이터를 집계(aggregation)하는 방법으로 데이터를 사용

# 금융에서 시계열 데이터를 다룰 때 down-sampling을 사용하는 경우가 더 많음.

import pandas as pd
index = pd.date_range(start= '2020-05-01', end= '2020-12-01', freq='B')
series = pd.Series(range(len(index)), index=index)
series

# 0~ 20 까지의 합 = 210, 이렇게 쭉 내려감. 
print(series.resample(rule='M').sum())
# arr = np.arange(21)
# print(arr)
# print(np.sum(arr))
print("------------------------------------------------------------------------")
# the last day of each month
print(series.resample(rule='M').last())
print("------------------------------------------------------------------------")
series.resample(rule='MS').first()

# rolling() func 과 resample() func 모두 일정 시간 간격으로 데이터를 조정. 차이점은?
# rolling()은 시간 기반 윈도우 작업을 수행하여 새로운 시점에 새로운 결과값을 계산
# resample()은 주기 기반 윈도우 작업을 수행하여 고정된 크기 내에서 일정 값을 리턴.

# open API for financial data analysis

# 금융 및 주가 데이터 가져오는 4가지 방법
# 1. 데이터 구매
# 2. 증권사 API 이용
# 3. 금융 웹 페이지 크롤링
# 4. 금융 데이터 제공 오픈 API 활용

# 종목 코드
# 한국 종목 코드
# KOSPI
# KOSDAQ
# KRX : KOSPI + KOSDAQ
# KONEX

# US 종목 코드
# NASDAQ
# NYSE
# AMEX
# SP500 

import FinanceDataReader as fdr
# 종목 코드 가져옴.
df_krx = fdr.StockListing('NYSE')
print(len(df_krx))
df_krx.head(8)

# read price data
df = fdr.DataReader('PLTR', '2021')
print(df.head(20))
df['Close'].plot()

# investing.com : 금융데이터를 조회할 수 있는 사이트로 전 세계를 대상으로 함.
# API 에서 제공하는 거래소 코드 - 한국어로 검색 가능.
exchange_map = {
    'KRX':'Seoul', '한국 거래소':'Seoul',
    'NASDAQ':'NASDAQ', '나스닥':'NASDAQ',
    'NYSE':'NYSE', '뉴욕증권거래소':'NYSE',
    'AMEX':'AMEX', '미국증권거래소':'AMEX',
    'SSE':'Shanghai', '상해':'Shanghai', '상하이':'Shanghai',
    'SZSE':'Shenzhen', '심천':'Shenzhen',
    'HKEX':'Hong Kong', '홍콩':'Hong Kong',
    'TSE':'Tokyo', '도쿄':'Tokyo'
}
# 캐논
jp_df = fdr.DataReader(symbol='7751',start='2019-01-01',exchange='TSE')
jp_df['Close'].plot()

# 철도 건설 공사
ch_df = fdr.DataReader(symbol='601186', start='2019-01-01',exchange='상해')
ch_df['Close'].plot()