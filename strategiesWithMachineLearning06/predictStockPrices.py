# 주가 방향 예측 using ETFs

import warnings
# warnings.filterwarnings('ignore')
import glob
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import svm
import seaborn as sns
sns.set()

df = pd.read_csv('../contents/dataForML/ETFs_main.csv')

# make 기술 지표
# 이동 평균
def moving_average(df, day=5):
    ma = pd.Series(df['CLOSE_SPY'].rolling(day, min_periods=day).mean(), name='MA_'+str(day))
    return df.join(ma)

# 거래량 이동 평균
def volume_moving_average(df, day=5):
    vma = pd.Series(df['VOLUME'].rolling(day, min_periods=day).mean(), name='VMA_'+str(day))
    return df.join(vma)

'''
Calculate Relative Strength Index(RSI)
df : pandas.DataFrame
n : 
return pandas.DataFrame
'''
# 시장 강도 지수
def relative_strength_index(df, day=14):
    i = 0
    upI = [0]
    doI = [0]
    while i + 1 <= df.index[-1]:
        upMove = df.loc[i + 1, 'HIGH'] - df.loc[i, 'HIGH']
        doMove = df.loc[i, 'LOW'] - df.loc[i + 1, 'LOW']
        if upMove > doMove and upMove > 0:
            upD = upMove
        else:
            upD = 0
        upI.append(upD)
        if doMove > upMove and doMove > 0:
            doD = doMove
        else:
            doD = 0
        doI.append(doD)
        i += 1
    upI = pd.Series(upI)
    doI = pd.Series(doI)
    posDI = pd.Series(upI.ewm(span=day, min_periods=day).mean())
    negDI = pd.Series(doI.ewm(span=day, min_periods=day).mean())
    rsi = pd.Series(posDI / (posDI + negDI), name='RSI_'+str(day))
    return df.join(rsi)

# 실제 경과 일수는 60일, 영업일 기준 45일
df = moving_average(df, day=45)
df = volume_moving_average(df, day=45)
# 14 or 21일을 주로 사용
df = relative_strength_index(df, day=14)
print(len(df))
print("=========================================================================\n")
df = df.set_index('Dates')
# 기술 지표를 만든 후 결측치 처리를 해줘야한다.
# 혹은 쿠션 데이터를 만들어서 내가 구하고 싶은 범위 값을 보존하는 방법이 있다. (이때에도 물론 쿠션 데이터 결측치 처리를 해줘야한다.)
df = df.dropna()
print(len(df))
print("=========================================================================\n")

df['target'] = df['CLOSE_SPY'].pct_change()
df['target'] = np.where(df['target'] > 0, 1, -1)
print(df['target'].value_counts())

# 당일까지의 데이터를 사용해 다음날을 예측하므로 shift() 함수로 다음날 트렌드(1 or -1)를 한 행 앞으로 당겨줌.
df['target'] = df['target'].shift(-1)

df = df.dropna()
print(len(df))
print("=========================================================================\n")

# target 변수 소수점 처리, label 변수 정리.
df['target'] = df['target'].astype(np.int64)
target_data = df['target']
train_data = df.drop(['target', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE_SPY'], axis=1)

print(train_data.head(5))
print("=========================================================================\n")

# 설명 변수(x_var) 데이터프레임
up = df[df['target']==1].target.count()
total = df.target.count()
print('up/down ratio : {0:.2f}'.format((up/total)))
print("=========================================================================\n")

# train a model
# shuffle=False 기간이 섞이지 않도록 해준다.(시계열 데이터를 다룰때 중요)
train_input, test_input, train_target, test_target = train_test_split(train_data, target_data, test_size=0.3, shuffle=False)

# 양성 샘플(up) 과 음성 샘플 비율이 비슷한지 확인
# 큰 차이가 날 수 있기 때문이다. (비율이 비슷해야한다)
train_count = train_target.count()
test_count = test_target.count()
print('train set label ratio')
print(train_target.value_counts()/train_count)
print('test set label ratio')
print(test_target.value_counts()/test_count)
print("=========================================================================\n")

# 정확도, ROC-AUC 점수 등 혼동 행렬을 계산하는 함수
def get_confusion_matrix(target, pred):
    confusion = confusion_matrix(target, pred)
    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)
    roc_score = roc_auc_score(target, pred)
    print('confusion matrix')
    print('accuracy:{0:.4f}, precision:{1:.4f}, recall:{2:.4f}, F1:{3:.4f}, ROC AUC score:{4:.4f}'.format(accuracy,precision,recall,f1,roc_score))

# XGBoost 분류기
xgb_dis = XGBClassifier(n_estimators=400, learning_rate=0.1,max_depth=3)
xgb_dis.fit(train_input, train_target)
xgb_pred = xgb_dis.predict(test_input)
print(xgb_dis.score(train_input, train_target))
get_confusion_matrix(test_target, xgb_pred)
print("=========================================================================\n")

n_estimators = range(10,200,10)
params = {
    'bootstrap': [True],
    'n_estimators': n_estimators,
    'max_depth': [4,6,8,10,12],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6, 8, 10],
    'max_features': [4]
}
# 교차 검증 방법
my_cv = TimeSeriesSplit(n_splits=5).split(train_input)
# GridSearchCV() : 여러 가지 값을 돌아가면서 테스트함
clf = GridSearchCV(RandomForestClassifier(), params, cv=my_cv, n_jobs=-1)
clf.fit(train_input, train_target)

print('best parameter:\n', clf.best_params_)
print('best prediction:{0:.4f}'.format(clf.best_score_))
print("=========================================================================\n")
# test set
pred_con = clf.predict(test_input)
accuracy_con = accuracy_score(test_target, pred_con)
print('accuracy: {0:.4f} '.format(accuracy_con))
get_confusion_matrix(test_target,pred_con)
print("=========================================================================\n")
# 전체적 통계를 확인
print(df['target'].describe())
print("=========================================================================\n")
# 0이 아니라 0.05 이상의 수익을 올렸을 때 상승추세라고 판단해서 다시 모델을 만들어본다.
df['target'] = np.where(df['target'] > 0.0005, 1, -1)
print(df['target'].value_counts())

df['target'] = df['target'].shift(-1)
df = df.dropna()
print(len(df))
df['target'] = df['target'].astype(np.int64)
target_data = df['target']
train_data = df.drop(['target', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE_SPY'], axis=1)
train_input, test_input, train_target, test_target = train_test_split(train_data, target_data, test_size=0.3, shuffle=False)
my_cv = TimeSeriesSplit(n_splits=5).split(train_input)
print("=========================================================================\n")
clf2 = GridSearchCV(RandomForestClassifier(), params, cv=my_cv, n_jobs=-1)
# 다시 훈련 후 확인
clf2.fit(train_input, train_target)
print('best parameter:\n', clf2.best_params_)
print('best prediction:{0:.4f}'.format(clf2.best_score_))
print("=========================================================================\n")
# 다시 test set
pred_con = clf2.predict(test_input)
accuracy_con = accuracy_score(test_target, pred_con)
print('accuracy: {0:.4f} '.format(accuracy_con))
get_confusion_matrix(test_target,pred_con)
print("=========================================================================\n")





















