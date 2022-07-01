'''
서울시 부동산 실거래가 정보 데이터를 기반으로 분석

<분석 방향 >

- 해당 데이터를 기반으로 부동산 가격을 예측 가능한지?
- 가장 높은 가격의 부동산이 있는 위치 확인
- 부동산 가격과 다른 변수 간의 상관관계 확인
- 부동산 가격에 대한 추세는 어떨 것인지 예상
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import BinaryEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from yellowbrick.regressor import ResidualsPlot

pd.set_option("display.max_columns", 50)

raw_data = pd.read_csv("/Users/yingying/ITWILL/팀프로젝트/최종프로젝트/서울시 부동산 실거래가 정보.csv",
                encoding="euc_kr", low_memory=False)


##################################
#### 1. 변수 탐색
##################################

# 1) raw_data 보기
raw_data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 640000 entries, 0 to 639999
Data columns (total 20 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   접수연도              640000 non-null  int64    
 1   자치구코드             640000 non-null  int64    - 삭제 대상
 2   자치구명              640000 non-null  object   
 3   법정동명              640000 non-null  object 
 4   법정동코드             640000 non-null  int64    - 삭제 대상
 5   지번구분명             591364 non-null  object   - 삭제 대상
 6   본번                591375 non-null  object     - 삭제 대상
 7   부번                591375 non-null  float64    - 삭제 대상 
 8   건물명               591363 non-null  object     - 삭제 대상 
 9   계약일               640000 non-null  int64      - 삭제 대상
 10  물건금액(만원)          640000 non-null  int64    
 11  건물면적(㎡)           640000 non-null  float64   
 12  토지면적(㎡)           480793 non-null  float64   - 삭제 대상
 13  층                 591406 non-null  float64
 14  권리구분              5976 non-null    object     - 삭제 대상
 15  취소일               15316 non-null   float64     - 삭제 대상
 16  건축년도              637484 non-null  float64    
 17  건물용도              640000 non-null  object 
 18  신고구분              46255 non-null   object      - 삭제 대상
 19  신고한 개업공인중개사 시군구명  34200 non-null   object  - 삭제 대상
dtypes: float64(6), int64(5), object(9)
'''

# 2) 컬럼명 수정
df = raw_data.rename(columns={'물건금액(만원)':'물건금액', '건물면적(㎡)':'건물면적'})

# 3) 필요한 칼럼만 추출하기
df = df.loc[:,['접수연도','자치구명','법정동명','물건금액','건물면적','층', '건축년도', '건물용도']]
df.info()
'''
#   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   접수연도    640000 non-null  int64    
 1   자치구명    640000 non-null  object   - 인코딩 필요
 2   법정동명    640000 non-null  object   - 인코딩 필요
 3   물건금액    640000 non-null  int64    
 4   건물면적    640000 non-null  float64  
 5   층       591406 non-null  float64
 6   건축년도    637484 non-null  float64  
 7   건물용도    640000 non-null  object   - 인코딩 필요
dtypes: float64(3), int64(2), object(3)
'''

# 4) 결측치 확인
df.isnull().sum()
'''
접수연도        0
자치구명        0
법정동명        0
물건금액        0
건물면적        0
층          48594
건축년도      2516
건물용도        0
dtype: int64
'''
# [해석] null 값이 있는 부동산들은 어떠한 부동산인지 추가 확인 필요

###############
## 추가 확인
###############
# '층'의 null값 확인
df[df['층'].isnull()]['건물용도'].unique()   # array(['단독다가구'], dtype=object)
# [해석] 층이 null로 된 부동산들은 모두 단독다가구이므로 null값을 1로 추후 대체 예정

# '건축년도'의 null값 확인
df[df['건축년도'].isnull()]['접수연도'].unique()
df[df['건축년도'].isnull()]['자치구명'].unique()
df[df['건축년도'].isnull()]['법정동명'].unique()
df[df['건축년도'].isnull()]['물건금액'].max()
df[df['건축년도'].isnull()]['물건금액'].min()
df[df['건축년도'].isnull()]['건물면적'].max()
df[df['건축년도'].isnull()]['건물면적'].min()
df[df['건축년도'].isnull()]['층'].max()
df[df['건축년도'].isnull()]['층'].min()
df[df['건축년도'].isnull()]['건물용도'].unique()
# [해석] 건축년도가 null로 된 부동산들은 속성 값이 다양하므로 대체하기 애매하니 추후 삭제 예정

# 5) 이상치 확인
# 범주형 변수의 경우
df['접수연도'].unique()   # 이상치 없음
df['자치구명'].unique()   # 이상치 없음
df['층'].unique()       # 이상치(층 = 0)
df['건물용도'].unique()   # 이상치 없음

# 연속형 변수의 경우
df.describe()
'''
          접수연도          물건금액          건물면적            층          건축년도 
count  640000.000000  6.400000e+05  640000.000000  591406.000000    637484.000000  
mean     2019.640220  6.029282e+04      71.161121       6.663972      1983.039118  
std         1.209938  6.799329e+04      67.255728       5.796018       193.321607  
min      2017.000000  1.700000e+03       5.070000      -3.000000         0.000000 
25%      2019.000000  2.400000e+04      39.480000       3.000000      1993.000000  
50%      2020.000000  4.000000e+04      59.490000       5.000000      2002.000000  
75%      2021.000000  7.490000e+04      84.800000      10.000000      2012.000000 
max      2022.000000  1.108778e+07    3619.840000      73.000000      2022.000000      
'''
# [해석] 건축년도 최소값 0 & 물건금액 최댓값 이상치 같으므로 추가 확인 필요

###############
## 추가 확인
###############
# 건축년도 0으로 된 부동산 확인
df[df['건축년도'] == 0]  # [5977 rows x 8 columns]
df[df['건축년도'] == 0]['접수연도'].unique()
df[df['건축년도'] == 0]['자치구명'].unique()
df[df['건축년도'] == 0]['법정동명'].unique()
df[df['건축년도'] == 0]['물건금액'].max()  # 950000
df[df['건축년도'] == 0]['물건금액'].mean()  # 95219.54810105404
df[df['건축년도'] == 0]['물건금액'].min()  # 11480
df[df['건축년도'] == 0]['건물면적'].max()  # 273.96
df[df['건축년도'] == 0]['건물면적'].min()  # 12.71
df[df['건축년도'] == 0]['층'].max()      # 73.0
df[df['건축년도'] == 0]['층'].min()      # 0.0
df[df['건축년도'] == 0]['건물용도'].unique()  # ['아파트', '단독다가구']
# [해석] 건축년도가 0으로 된 부동산들은 속성 값이 다양하므로 대체하기 애매하니 추후 삭제 예정

# 물건금액 최댓값 확인 전에 먼저 정상범주를 계산해 보기
# 방벙1) maxval1 = 151250.0
maxval1 = df['물건금액'].describe()['75%'] + (df['물건금액'].describe()['75%'] - df['물건금액'].describe()['25%']) * 1.5
# 방법2) maxval2 = 264272.6734560396
maxval2 = df['물건금액'].describe()['mean'] + 3*df['물건금액'].describe()['std']

# 물건금액 최댓값 확인
df.sort_values(by="물건금액",ascending=False).head()
'''
       접수연도 자치구명   법정동명  물건금액   건물면적   층   건축년도  건물용도
24796   2022  용산구    한남동  11087780  1742.90 NaN  1970.0  단독다가구
30280   2022  용산구    한남동  11087780  1742.90 NaN  1970.0  단독다가구
34576   2022  강남구    역삼동   3000000  2536.72 NaN  2007.0  단독다가구
35565   2022  성동구  성수동1가   3000000  1494.00 NaN  1984.0  단독다가구
245915  2020  서초구    서초동   2900000  2804.97 NaN  1991.0  단독다가구
'''
# [해석] 행(24796 & 30280)의 물건금액이 최댓값으로 확인됨. 정상범주에 벗아났으므로 추후 삭제 예정


##################################
#### 2. 데이터 전처리
##################################

# 1) 결측치 처리
df['층'] = df['층'].fillna(1)  # 층의 null값은 1로 대체
df = df.dropna()  # 2516 rows(건축년도=null) 삭제됨
df.isnull().sum()  # 결측치 없음

# 2) 이상치 처리
df = df[df['건축년도'] > 0]  # 5977 rows(건축년도=0) 삭제됨
df = df.drop([24796, 30280], axis=0)  # 물건금액 최댓값 row 삭제됨

# 3) 자치구명 순으로 데이터셋 정렬
df = df.sort_values(by='자치구명', ascending=True)


########################################
#### 3. 데이터 시각화 및 분석
########################################

sns.set(font="AppleGothic",
            rc={"axes.unicode_minus":False}, style="darkgrid")

#####################
## 범주형 vs 연속형
#####################

# 1) 자치구명 vs 물건금액
# 1-1) 자치구별 거래량
plt.figure(figsize=(20, 7))
sns.countplot(df['자치구명'])
plt.xticks(rotation=55)
plt.title('자치구별 거래량')
plt.show()
df['자치구명'].value_counts()
# [해석] 자치구별 거래량 top3 : 강서구, 은평구, 송파구

# 1) 자치구명 vs 물건금액
group_gu = df.groupby('자치구명')
group_gu_price = group_gu['물건금액'].mean()

plt.figure(figsize=(20, 7))
sns.lineplot(data=group_gu_price)
plt.xticks(rotation=55)
plt.title('자치구명별 물건금액 평균')
plt.show()
# [해석] 자치구별 부동산 거래가격 top3: 강남구, 서초구, 용산구

# 2) 법정동명 vs 물건금액
group_dong = df.groupby('법정동명')
group_dong_price = group_dong['물건금액'].mean()
group_dong_price = group_dong_price.sort_values(ascending=False)

plt.figure(figsize=(20, 7))
sns.lineplot(data=group_dong_price[:30])
plt.xticks(rotation=55)
plt.title('법정동명별 물건금액 평균 top30')
plt.show()
# [해석] 법정동명별 부상동 거래가격 top3: 봉익동(종로구), 수표동(중구), 명동1가(중구)

# 3) 건물용도 vs 물건금액
plt.figure(figsize=(20, 7))
sns.scatterplot(data=df,x="건물용도",y="물건금액")
plt.title("건물용도와 물건금액의 관계")
plt.show()
df['건물용도'].value_counts()
'''
아파트      290943
연립다세대    229303
오피스텔      62717
단독다가구     48542
Name: 건물용도, dtype: int64
'''
# [해석] 단독다가구 개수는 다른 용도에 비하면 적지만, 거래가격은 상대적으로 높은 편

#####################
## 연속형 vs 연속형
#####################

# 4) 접수연도 vs 물건금액
group_year1 = df.groupby('접수연도')
group_year1_price = group_year1['물건금액'].mean()

plt.figure(figsize=(20, 7))
sns.lineplot(data=group_year1_price)
plt.xlabel('접수연도')
plt.ylabel('물건금액')
plt.title('접수연도별 물건금액 평균')
plt.show()
# [해석] 2017년부터 2021년까지 부동산 거래가격이 상승하다가 2021년부터 현재까지 떨어지고 있는 추세임

# 5) 건물면적 vs 물건금액
plt.figure(figsize=(20,7))
sns.scatterplot(x='건물면적', y='물건금액', data=df)
plt.title("건물면적과 물건금액의 관계")
plt.show()
# [해석] 건물면적과 부동산 거래가격 간에 유의미한 상관성이 없음

# 6) 층 vs 물건금액
plt.figure(figsize=(20,7))
sns.scatterplot(data=df,x="층",y="물건금액")
plt.title("층수와 물건금액의 관계")
plt.show()
df['층'].value_counts().head()
'''
1.0    92724
2.0    80813
3.0    76032
4.0    65998
5.0    51319
'''
# [해석] 층수에 따라 부동산 거래가격이 변화 없음, 가격이 높은 층은 대부분 1층(단독주택)임

# 7) 건축년도 vs 물건금액
plt.figure(figsize=(20, 7))
sns.scatterplot(data=df,x="건축년도",y="물건금액")
plt.title('건축년도와 물건금액의 관계')
plt.show()
# [해석] 건축년도에 따른 부동산가격에 대한 지표를 보는 것은 의미가 없음
# 왜냐하면 1950년대 한국전쟁도 있었고, 1960년대부터 아파트 공급이 시작되었고, 또한 입지별로 가격차이가 다르기 때문임
# 신축 건물이라고 가격이 비싼건 아님

# 연속형 변수를 대상으로 상관계수 확인 및 시각화
corr = df.corr(method = 'pearson')
'''            
          접수연도    물건금액   건물면적    층     건축년도
접수연도  1.000000 -0.011696 -0.056327 -0.056069  0.140506
물건금액 -0.011696  1.000000  0.645417  0.168094 -0.179911
건물면적 -0.056327  0.645417  1.000000 -0.046254 -0.254733
층       -0.056069  0.168094 -0.046254  1.000000  0.168133
건축년도  0.140506 -0.179911 -0.254733  0.168133  1.000000
'''
heat = sns.heatmap(corr, cbar= True, annot=True,
                   annot_kws ={'size':20}, fmt='.2f',
                   square = True, cmap = 'Blues')


##################################
#### 4. feature engineering
##################################

# 1) 인코딩
df_encoded = BinaryEncoder(cols=['자치구명', '법정동명', '건물용도']).fit_transform(df)
df_encoded.shape  # (631505, 22)

# 2) 스케일링
# 2-1) subset로 X변수 생성 -> X변수 스케일링
X = df_encoded.drop(columns='물건금액')  
X.shape  # (631505, 21)

X_not_scaled = X.drop(columns=['접수연도', '건물면적', '층', '건축년도'])
X_not_scaled.reset_index(drop=True, inplace=True)
X_scaled = scale(X[['접수연도', '건물면적', '층', '건축년도']])
X_scaled = pd.DataFrame(X_scaled, columns=['접수연도', '건물면적', '층', '건축년도'])
X_new = pd.concat(objs=[X_not_scaled,X_scaled], axis = 1)
X_new.shape  # (631505, 21)

# 2-2) subset로 y변수 생성  -> y변수 로그변환
y = df_encoded['물건금액']
y.shape  # (631505,)
y_log = np.log(y)

# 3) train/test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_log, test_size=0.3, random_state=42)

print("X Train:", X_train.shape)  # X Train: (442053, 21)
print("X_test:", X_test.shape)    # X_test: (189452, 21)
print("y_train:", y_train.shape)  # y_train: (442053,)
print("y_test:", y_test.shape)    # y_test: (189452,)


##################################
#### 5. 예측 모델 생성, 평가 및 시각화
##################################

LR = LinearRegression()
DTR = DecisionTreeRegressor(random_state=42)
RFR = RandomForestRegressor(random_state=42)
KNR = KNeighborsRegressor()
XGB = XGBRegressor(random_state=42)
LGB = lgb.LGBMRegressor(random_state=42)

li = [LR, DTR, RFR, KNR, XGB, LGB]
d = {}
for i in li:
    model = i.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(i, ":", r2_score(y_test, y_pred)*100)
    print(i, ":", mean_squared_error(y_test, y_pred))
    d.update({str(i):i.score(X_test, y_test)*100})
'''
LinearRegression() : 61.21873279835921
LinearRegression() : 0.2478906454911216
DecisionTreeRegressor(random_state=42) : 88.73676777187892
DecisionTreeRegressor(random_state=42) : 0.0719948085457922
RandomForestRegressor(random_state=42) : 93.57277913552876
RandomForestRegressor(random_state=42) : 0.041082925952981
KNeighborsRegressor() : 90.80217378996667
KNeighborsRegressor() : 0.05879269144211598
XGBRegressor(random_state=42) : 90.67247429588016
XGBRegressor(random_state=42) : 0.05962173323545928
LGBMRegressor(random_state=42) : 88.29748387949567
LGBMRegressor(random_state=42) : 0.0748027200838692
'''

# 모델 r2 score 비교 시각화
x_label = ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor',
           'KNeighborsRegressor', 'XGBRegressor', 'LGBMRegressor']
plt.figure(figsize=(20, 7))
r2 = list(d.values())
ax = sns.barplot(x_label, r2, palette='pastel')
ax.bar_label(ax.containers[0])
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Comparing R2 Score of Models')


##################################
#### 6. 모델 튜닝
##################################

# 1) GridSearchCV 모델 생성
model_RFR = RFR.fit(X_train, y_train)

parmas = {'n_estimators' : [100, 200],
          'max_depth' : [None, 5],
          'min_samples_split' : [2, 6],
          'min_samples_leaf' : [1, 6]} # dict 정의

grid_model = GridSearchCV(model_RFR, param_grid=parmas,
                          scoring='neg_mean_squared_error', cv=3, n_jobs=-1)  # neg_mean_squared_error

grid_model = grid_model.fit(X_train, y_train)


# 2) Best score & parameters
print('best score =', grid_model.best_score_)  # best score = -0.04422525910036088

print('best parameters =', grid_model.best_params_)
# best parameters = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

# 3) Best parameters 적용하여 모델 만들기
best_model = RandomForestRegressor(random_state=42, n_estimators=200).fit(X_train, y_train)
score = best_model.score(X_test, y_test)
print('best score =', score*100)  # best score = 93.60748540518264

# 4) over-fitting 유무 확인 및 시각화
train_score = best_model.score(X_train, y_train)
print('train score:', train_score)  # train score: 0.9898107495063089

test_score = best_model.score(X_test, y_test)
print('test score:', test_score)    # test score: 0.9360748540518264

# 시각화
visualizer = ResidualsPlot(best_model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
# [해석] 혼련셋의 결정계수와 테스트셋의 결과계수를 비교해 보니, 0.05의 차이로 over-fitting이라고 볼 수 없음

# 5) real value vs predicted value - before/after GridSearchCV

# 5-1) before GridSearchCV
rf_rgs = RandomForestRegressor(random_state=42).fit(X_train , y_train)
y_pred1 = rf_rgs.predict(X=X_test)
y_true = y_test

# 그래프
plt.scatter(y_true, y_pred1, c = "green")
plt.title("RandomForestRegressor")
plt.xlabel('real values')
plt.ylabel('pred values')
plt.axis('equal')
plt.axis('square')
plt.xlim([6.5, plt.xlim()[1]])
plt.ylim([6.5, plt.ylim()[1]])
_= plt.plot([-100, 100], [-100, 100], c = "green")

# 5-2) after GridSearchCV
y_pred2 = best_model.predict(X=X_test)
y_true2 = y_test

# 그래프
plt.scatter(y_true2, y_pred2)
plt.title("RandomForest GridSearch")
plt.xlabel('real values')
plt.ylabel('pred values')
plt.axis('equal')
plt.axis('square')
plt.xlim([6.5, plt.xlim()[1]])
plt.ylim([6.5, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


##################################
#### 7. 딥러닝(DNN)
##################################

import tensorflow as tf
import numpy as np
import random as rd
from keras import Sequential
from keras.layers import Dense

tf.random.set_seed(42)
np.random.seed(42)
rd.seed(42)

# DNN model layer 구축
model = Sequential()
model.add(Dense(256, input_shape=(21,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

# train set / test set / val set split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_log, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# model 학습과정 설정
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# model training
model_fit = model.fit(x=X_train, y=y_train,
                      epochs=30, verbose=1,
                      batch_size=20000, validation_data=(X_val, y_val))

# model evaluation/prediction
loss, mse = model.evaluate(X_test, y_test)
print('mse : ', mse)  # mse :  0.0881928876042366

y_predict = model.predict(X_test)
r2_y_predict = r2_score(y_test, y_predict)
print('R2 :', r2_y_predict)   # R2 : 0.8618141660484626
# [해석] Randomforest model의 mse는 0.04 이고 r2 score는 93.57인 반면,
# DNN model의 mse는 0.08이고 r2 score는 86.18로
# 예상과는 달리 Randomforest Model의 예측력이 더 좋음

# loss vs val_loss
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()
# [해석] epochs 5부터 loss rate의 차이가 거의 없음


##################################
#### 8. 결론
##################################
'''
1) 해당 데이터를 기반으로 모델을 생성해 부동산 가격을 예측할 수 있었음.

2) 가장 높은 가격의 부동산의 위치는 자치구별로는 강남구, 서초구, 용산구 순으로 나타났고, 
   법정동별로는 봉익동(종로구), 수표동, 명동1가(중구) 순으로 높게 나타났음.

3) 부동산 가격과 가장 높은 상관관계를 나타난 변수는 건물면적, 자치구명, 건물용도로 나타났음

4) 부동산 가격은 2017년부터 2021년 상반기까지는 상승하다가 2021년부터 현재까지는 하락하고 있는 추세임
'''
