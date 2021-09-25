import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pandas_datareader import data as pdr
pd.set_option('display.max_rows',12)
pd.set_option('display.max_columns',12)

# k-nearest neighbor(KNN)
# 유한한 특성을 가진 데이터 사이의 거리는 가깝다.
# 이 때, 어떤 분포도 가정하지 않기 때문에 KNN은 비모수적 방법(non-parametric method)에 속함
# 비모수적 방법에서는 데이터가 많아야 분산이 작아지므로, 데이터를 많이 확보해야한다.
# 단점은 설명 변수가 많아진다면,(차원이 높아지는 경우) 연산 비용이 높아진다.
# 또 다른 단점은 k 값과 사용할 거리 척도 방법을 적절하게 설정해야 함.(x,y값의 기준값을 똑같이 맞춰줘야함)

df = pdr.get_data_yahoo('SPY', '2012-01-01','2017-01-01')
df = df.dropna()
print(df.head(3))
print("=========================================================================\n")

tmp_df = df[['Open','High','Low','Close']].copy()
print(tmp_df.head(3))
print("=========================================================================\n")

# preprocessing
tmp_df['Open-Close'] = tmp_df['Open'] - tmp_df['Close']
tmp_df['High-Low'] = tmp_df['High'] - tmp_df['Low']
tmp_df = tmp_df.dropna()
input_data = tmp_df[['Open-Close','High-Low']]
knn_target = lambda df : df['Close'].shift(-1) > df['Open'].shift(-1)
input_target = knn_target(tmp_df)
print(input_data.head(2))
print(input_target.head(3))
print("=========================================================================\n")
# dividie train set and test set
split_percentage = 0.7
split = int(split_percentage * len(tmp_df))
train_input = input_data[:split]
train_target = input_target[:split]

test_input = input_data[split:]
test_target = input_target[split:]

# train model
train_acc = []
test_acc = []

for k in range(1,15):
    clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=k)
    clf.fit(train_input, train_target)
    prediction = clf.predict(test_input)
    train_acc.append(clf.score(train_input,train_target))
    test_acc.append((prediction==test_target).mean())

# graph
plt.figure(figsize=(12,9))
plt.plot(range(1,15), train_acc, label='Train set')
plt.plot(range(1,15), test_acc, label='Test set')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.xticks(np.arange(0, 16, step=1))
plt.legend()
plt.show()
print("=========================================================================\n")

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_input, train_target)

accuracy_train = accuracy_score(train_target, knn.predict(train_input))
accuracy_test = accuracy_score(test_target, knn.predict(test_input))
print('훈련 정확도 : %.2f' %accuracy_train)
print('test accuracy : %.2f' %accuracy_test)
print("=========================================================================\n")
# calc return
tmp_df['Predicted_Signal'] = knn.predict(input_data)

tmp_df['SPY_ret'] = np.log(tmp_df['Close'] / tmp_df['Close'].shift(1))
cum_spy_ret = tmp_df[split:]['SPY_ret'].cumsum() * 100

tmp_df['st_ret'] = tmp_df['SPY_ret'] * tmp_df['Predicted_Signal'].shift(1)
cum_st_ret = tmp_df[split:]['st_ret'].cumsum() * 100

plt.figure(figsize=(10,5))
plt.plot(cum_spy_ret, color='r', label='spy ret')
plt.plot(cum_st_ret, color='g', label='st ret')
plt.legend()
plt.show()

# sharpe ratio
std = cum_st_ret.std()
sharpe = (cum_st_ret - cum_spy_ret) / std
sharpe = sharpe.mean()
print('sharpe ratio : %.2f' %sharpe)
print("=========================================================================\n")

