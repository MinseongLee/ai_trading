import sys

import pandas as pd
import pandas_datareader as pdr
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# clustering algorithm
# 뭔가를 기준으로 군집화하는 것.
# 기준의 예로는 섹터별, 변동성, 가격의 움직임 등이 될 수 있다.

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500_url)
tickers = data_table[0]['Symbol'].tolist()
tickers = tickers[0:60]
security = data_table[0]['Security'].tolist()[0:60]
sector = data_table[0]['GICS Sector'].tolist()[0:60]
print(len(tickers))
print(len(security))
print(len(sector))
print("=========================================================================\n")

prices_list = []
# load data
for ticker in tickers:
    try:
        prices = pdr.DataReader(ticker, 'yahoo', '01/01/2018')['Adj Close']
        prices = pd.DataFrame(prices)
        prices.columns = [ticker]
        prices_list.append(prices)
    except Exception as e:
        raise
    prices_df = pd.concat(prices_list,axis=1)
prices_df.sort_index(inplace=True)
print(prices_df.head(4))
print("=========================================================================\n")
# 수익률 변화의 정도를 계산하고 전치를 사용해 티커명을 Index로 사용
df = prices_df.pct_change().iloc[1:].T
print(df.head(4))
print("=========================================================================\n")
# 회사명 리스트와 값의 리스트를 저장
companies = list(df.index)
movements = df.values

# data normalization(데이터 정규화)
normalize = Normalizer()
array_norm = normalize.fit_transform(df)
df_norm = pd.DataFrame(array_norm, columns=df.columns)
final_df = df_norm.set_index(df.index)
print(final_df.head(10))
print("=========================================================================\n")
# 누락된 데이터가 없는지 확인
col_mask = df.isnull().any(axis=0)
row_mask = df.isnull().any(axis=1)
df.loc[row_mask, col_mask]
print("=========================================================================\n")
# clustering
num_of_clusters = range(2,12)
error = []

for num_clusters in num_of_clusters:
    clusters = KMeans(num_clusters)
    clusters.fit(final_df)
    error.append(clusters.inertia_/100)

table = pd.DataFrame({"Cluster_Numbers":num_of_clusters, "Error_Term":error})
print(table)
print("=========================================================================\n")
# 엘보(elbow) 방법으로 최적의 클러스터링 개수 찾기
plt.figure(figsize=(15,10))
plt.plot(table.Cluster_Numbers, table.Error_Term, marker="D", color='red')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()
print("=========================================================================\n")
# 클러스터가 잘 정의되지 않은 데이터에서는 엘보가 잘 보이지 않음
# 그러므로 임의로 클러스터 개수를 선택
clusters = KMeans(n_clusters=7)
clusters.fit(final_df)
print(clusters.labels_)

labels = clusters.predict(movements)
print(labels)
print("=========================================================================\n")
clustered_result = pd.DataFrame({'labels': labels, 'tickers': companies,
                                 'full-name':security, 'sector':sector})
clustered_result.sort_values('labels')
final_df['Cluster'] = clusters.labels_

plt.figure(figsize=(12,6))
sns.countplot(x = 'Cluster', data = final_df, palette = 'magma')
plt.title('Cluster_count')
plt.show()
plt.savefig('cluster_count.png', dpi=300)
print("=========================================================================\n")

