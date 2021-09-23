# 가치 투자를 위한 마법공식

import FinanceDataReader as fdr
import pandas as pd
import numpy as np

krx_df = fdr.StockListing('KRX')
print(krx_df.head(5))
# encoding= UTF-16 US-ASCII ISO-8859-1
# ksc5601 : 한국 산업 규격으로 지정된 인코딩
df = pd.read_csv('../contents/indicators/PER_ROA.csv',engine='python', encoding='ksc5601')
print(df.head(5))
print("=========================================================================\n")
# 여기서 NaN인 데이터는 제외하고 정리한다.
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
print(df.head(5))
print("=========================================================================\n")

'''
description : 특정 지표값 정리
parameter
sample : pandas Series
- 정렬할 데이터
sort : bool
- True : ASC
- False : DESC
standard : int
- 조건에 맞는 값을 True로 대체하기 위한 기준값
return pandas Series
- 정렬된 데이터
'''
def sort_sample(sample, sort = True, standard = 0):
    # 지표별 기준값 미만은 필터링
    sample_mask = sample.mask(sample < standard, np.nan)
    # 필터링된 종목에서 순위를 선정
    sample_mask_rank = sample_mask.rank(ascending=sort, na_option='bottom')
    return sample_mask_rank

per = pd.to_numeric(df['PER'])
roa = pd.to_numeric(df['ROA'])

# PER 지표값을 기준으로 순위 정렬 및 0 미만 값 제거
per_rank = sort_sample(per, sort=True, standard=0)
# ROA 지푯값을 기준으로 순위 정렬 및 0 미만 값 제거
roa_rank = sort_sample(roa, sort=False, standard=0)
print(per_rank.head(5))
print("=========================================================================\n")
print(roa_rank.head(5))
print("=========================================================================\n")

result_rank = per_rank + roa_rank
result_rank = sort_sample(result_rank, sort=True)
# where은 조건식에 맞는 애들을 원본 데이터로 가져오고 조건식에 맞지 않는 애들을 변경해줌.
result_rank = result_rank.where(result_rank <= 10, 0)
# mask는 조건식에 맞는 애들을 변경해줌.
result_rank = result_rank.mask(result_rank > 0, 1)

print(result_rank.head(5))
print("=========================================================================\n")
print(result_rank.sum())
print("=========================================================================\n")
# 마법공식을 통해 선정된 종목들 확인.
mf_df = df.loc[result_rank > 0,['종목명','시가총액']].copy()
# 선택된 종목명 추출
mf_stock_list = df.loc[result_rank > 0, '종목명'].values
print(mf_df)
print("=========================================================================\n")
# 종목 코드 추가
mf_df['종목 코드'] = ''
for stock in mf_stock_list:
    mf_df.loc[mf_df['종목명'] == stock,'종목 코드'] = krx_df[krx_df['Name'] == stock]['Symbol'].values
print(mf_df)
print("=========================================================================\n")
# 2019 마법공식 종목 수익률
mf_df['2019_return'] = ''
for x in mf_df['종목 코드'].values:
    # 개발 종목 가격 데이터 호출
    df = fdr.DataReader(x, '2019-01-01','2019-12-31')
    cum_ret = df.loc[df.index[-1], 'Close'] / df.loc[df.index[0],'Close'] -1
    # 2019 누적 수익률
    mf_df.loc[mf_df['종목 코드'] == x, '2019_return'] = cum_ret
    df = None
# 종목 당 누적수익률
print(mf_df)

print("=========================================================================\n")

# 날짜당 수익률
for ind, val in enumerate(mf_df['종목 코드'].values):
    # 가독성 위해 종목명 추출
    code_name = mf_df.loc[mf_df['종목 코드'] == val,'종목명'].values[0]
    print(val, code_name)
    df = fdr.DataReader(val, '2019-01-01','2019-12-31')
    if ind == 0:
        mf_df_rtn = pd.DataFrame(index=df.index)
    # periods 차이만큼 변동률 계산
    df['daily_rtn'] = df['Close'].pct_change(periods=1)
    # 누적 곱 계산
    df['cum_rtn'] = (1+df['daily_rtn']).cumprod()
    tmp = df.loc[:,['cum_rtn']].rename(columns={'cum_rtn':code_name})
    # 가독성을 위한 컬럼명 변경
    mf_df_rtn = mf_df_rtn.join(tmp,how='left') # 새로 계산된 누적 수익률 추가
    df = None

print("=========================================================================\n")
print(mf_df_rtn.tail(5))