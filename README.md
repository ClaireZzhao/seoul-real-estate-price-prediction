# Seoul-Real-Estate-price-prediction

## 서울 부동산 가격 예측 모델링

### 주제 선정
극심하게 변하는 서울 부동산 시장 속에서 어떠한 요인이 가격에 영향을 미치는지 알아보고자 한다.
어떤 모델이 가격 예측에 가장 높은 정확도를 보이는지 적용해 보고자 한다.

### 데이터 수집 및 전처리
데이터 출처: 서울 열린데이터 광장 - 서울시 부동산 실거래가 정보
- 2017년부터 2022년 상반기까지의 부동산 실거래가 정보 조회
- 20개의 칼럼 중 8개의 칼럼만 추출
- 결측치 및 이상치 확인 & 처리

### [데이터 시각화 및 분석](https://github.com/ClaireZzhao/seoul-real-estate-price-prediction/blob/main/resources/data%20visualization.pdf)

### 예측 모델
- [인코딩 및 스케일링](https://github.com/ClaireZzhao/seoul-real-estate-price-prediction/blob/8a62e6e14cdaf0c69dcaf34618644284663d3d01/seoul-real-estate-price-prediction.py#L326)
- [6가지의 예측 모델 생성 & 평가](https://github.com/ClaireZzhao/seoul-real-estate-price-prediction/blob/4bf9ea37b83051c2a92a9a19dc4ad642d82a5f89/seoul-real-estate-price-prediction.py#L356)
- [모델 튜닝](https://github.com/ClaireZzhao/seoul-real-estate-price-prediction/blob/9afb183c5e18168d9cf073ab7a318e8bfc36ddf4/seoul-real-estate-price-prediction.py#L402)
- [overfitting 유무 확인](https://github.com/ClaireZzhao/seoul-real-estate-price-prediction/blob/54b7afc85e0e036ccb449f12e9aba5de8968f3a6/seoul-real-estate-price-prediction.py#L431)


### 회고 및 고찰
#### 결론
- 총 6가지 모델 중 RandomForestRegressor 모델의 예측력이 약 94%로 가장 높음.
- 가장 높은 가격의 부동산의 위치는 자치구별로는 강남구, 서초구, 용산구 순으로 나타났고, 법정동별 로는 봉익동(종로구), 수표동, 명동1가(중구) 순으로 높게 나타났음.
- 부동산 가격과 가장 높은 상관관계를 나타난 변수는 건물면적, 자치구명, 건물용도로 나타났음.
- 부동산 가격은 2017년부터 2021년 상반기까지는 상승하다가 2021년부터 현재까지는 하락하고 있는 추세임.

#### 한계점
- 부동산 가격에 영향을 줄 수 있는 데이터셋에 포함되어 있지 않은 방의 개수나 편의시설 등과 같은 변수들이 있는데, 이러한 정보들이 있었으면 모델의 예측능력을 더욱 향상시킬 수 있었을 텐데 그러지 못해 아쉬움.
- 그리드 서치와 랜덤 서치를 이용하여 최적의 파라미터를 구하기 위해선 충분한 시간과 반복적인 트레이닝이 필요한데 데이터셋의 용량이 큼과 동시에 장비가 이러한 과정을 받쳐주지 못하여 트레이닝하는데 절대적인 시간이 부족하여 모델 예측력에 큰 변화는 없었음.
