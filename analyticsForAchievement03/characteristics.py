import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../contents/AAPL.csv', index_col='Date', parse_dates=['Date'])
price_df = df.loc[:,['Adj Close']].copy()
# 일별 수익률
price_df['daily_rtn'] = price_df['Adj Close'].pct_change()
# 누적 곱
price_df['st_rtn'] = (1+price_df['daily_rtn']).cumprod()
# 투자 성과 분석 지표들

# 연평균 복리 수익률(CAGR)(Compound Annual Growth Rate)
# CAGR formula
# https://www.investopedia.com/terms/c/cagr.asp

# CAGR formula with python
# 마지막 일자에 최종 누적 수익률의 누적 연도 제곱근을 구하는 것.
# 일(daily) 데이터를 사용했으므로 전체 연도를 구하려면,
# 전체 데이터 기간(p_df.index)을 252(1년 동안 주식 영업일)로 나눈 역수를 제곱(a**b) 연산한 후,
# -1을 하면 수익률을 구할 수 있다.
cagr = (price_df.loc['2021-09-17','st_rtn'] ** (252./len(price_df.index))) - 1
print("cagr = ", cagr)
print("cagr = ", round(cagr*100,2), '%')


# 산술평균(arithmetic mean) : N개의 변수를 모두 합한 후 N으로 나눈 값
# am = (x1+x2+,,,+xn)/n
# 기하평균(geometric mean) : N개의 변수를 모두 곱한 값의 N제곱근
# gm = (x1*x2,,,*xn)^1/n

# 최대 낙폭(MDD)(Maximum Draw Down)
# 최대 낙폭 지수, 투자 기간에 고점부터 저점까지 떨어진 낙폭 중 최댓값을 의미
# 낮을 수록 좋음
# MDD Formula
# mdd = (TV - PV) / PV
# TV(Trough Value) : 관측 기간 최저점 가격
# PV(Peak Value) : 관측 기간 최고점 가격
historical_max = price_df['Adj Close'].cummax()
daily_drawdown = price_df['Adj Close'] / historical_max - 1.0
historical_dd = daily_drawdown.cummin()
print("mdd = ", historical_dd.min())
print("mdd = ", round(-1*historical_dd.min()*100,2), '%')
plt.plot(historical_dd)
plt.xlabel('Date')
plt.ylabel('mdd')
plt.show()
# explanation
# 수정 종가(adj close)에 cummax() func 저장.
# cummax()는 누적 최댓값 return, 전체의 최댓값이 아닌 row별 차례로 진행하면서 누적 값 갱신
# 현재 수정 종가에서 누적 최댓값 대비 낙폭률을 계산 후
# cummin() func 사용해 최대 하락률 계산

# 변동성(Vol)(Valaility)
# 변동성에 있는 여러 가지 종류 중 여기에서는 주가 변화 수익률 관점의 변동성을 확인
# 금융 자산의 등락의 불확실성에 대한 예상 지표
# 수익률의 표준 편차(o)를 변동성으로 계산
# Vol formula
# o_p = o_일 * 루트P
# ex) 1년 252일, 일별 수익률의 표준 편차가 0.01
# o_연간 = 0.01 * 루트252 = 0.1587

# 이 공식은 일 단위 변동성을 의미.
vol = np.std(price_df['daily_rtn']) * np.sqrt(252.)
print("vol = ",vol)
print("vol = ", round(vol*100,2),'%')


# 샤프 지수(Sharpe ratio)
# 위험 대비 수익성 지표
# R_a : 자산 수익률, R_b : 무위험 수익률 or 기준 지표 자산 수익률
# 사후적 샤프 비율(ex-post Sharpe ratio)을 사용 : 자산의 실현 수익률을 사용하는 공식
# 실현 수익률의 산술평균 / 실현 수익률의 변동성 ( S_a = 첫번째 공식 사용 )
# Sharpe ratio Formula
# S_a = E[R_a - R_b] / o_a = E[R_a - R_b] / 루트var[R_a - R_b]

sharpe = np.mean(price_df['daily_rtn']) / np.std(price_df['daily_rtn']) * np.sqrt(252.)
print("sharpe = ", sharpe)
print("sharpe = ", round(sharpe*100,2),'%')

# apple result
# 연평균 복리 수익률(cagr)은 31.77%
# 최대 낙폭(mdd)은 81.8% 로.. 이 정도까지도 떨어진 경우가 존재한다. 애플조차도
# 변동성(vol)은 43.48%로 출렁거렸다는 의미
# 샤프 지수는 보통 1 이상만 되어도 좋다고 판단.