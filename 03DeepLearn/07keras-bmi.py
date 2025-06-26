from tensorflow.keras import Sequential          # 층을 순서대로 쌓는 모델
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd, numpy as np
'''
✅ 전체 요약 흐름
데이터 로딩 및 정규화
레이블 원-핫 인코딩 (enumerate 활용)
모델 구성 (layers 리스트로)
학습 설정 (EarlyStopping 포함)
평가
'''

# BMI 데이터 로드한 후 데이터프레임으로 변환
csv = pd.read_csv("./resData/bmi.csv")

# 데이터 정규화
# 몸무게와 키 데이터를 0~1사이의 값을 정규화한다.
csv["weight"] = csv["weight"] / 100
csv["height"] = csv["height"] / 200
# 입력데이터로 사용. 정규화된 몸무게와 키 컬럼을 추출
X = csv[["weight", "height"]]

# 레이블 변환. 원-핫 인코딩 형식의 딕셔너리 생성.
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
# 2만개의 레이블 데이터를 bclass 형식으로 변환.
'''np.empty((20000, 3))은 20000행 × 3열짜리 배열을 만드는데,
값을 0으로 채우지 않고 "쓰레기값"이 들어 있는 상태로 만듭니다.'''
y = np.empty((20000,3))
'''
enumerate()
파이썬에서 리스트 같은 걸 반복할 때, 인덱스(번호)와 값을 동시에 꺼내주는 함수.
=>	(인덱스, 값) 형태로 순회 가능
enumerate(..., start=1)	인덱스를 1부터 시작할 수도 있음

**y[0], y[1], ..., y[19999]**에 각 원-핫 벡터를 하나씩 넣는 것
'''
for i, v in enumerate(csv["label"]):
    y[i] = bclass[v]

X_train, y_train = X[1:15001], y[1:15001]
X_test, y_test = X[15001:20001], y[15001:20001]

# 배열 형식으로 레이어 정의 후 모델 구조 정의
# 각 층을 리스트에 추가하는 형식으로 정의(add함수 사용과 동일)
'''
🔍 모델 구성 (리스트로 레이어 정의)
model.add(...) 대신 리스트로 한 번에 정의
Dense(512) = 노드 512개인 완전연결층
Dropout(0.1) = 학습 중 10%의 노드를 무작위로 꺼서 과적합 방지
마지막 Activation('softmax') = 확률처럼 출력
'''
layers = [
    Dense(512, input_shape=(2,)),
    Activation('relu'),
    Dropout(0.1),
    Dense(512),
    Activation('softmax')
]
model = Sequential(layers)

# 모델 구축. RMSprop 옵티마이저 사용. 손실함수 적용.
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = "rmsprop",
    metrics = ['accuracy']
)

# 데이터 훈련
hist = model.fit(
    X_train, y_train,
    batch_size=100, # 한번에 100개씩 묶어서 학습
    epochs=20, # 최대 2개번의 반복학습
    validation_split=0.1, # 훈련 데이터 중 10%를 검증 데이터로 사용
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1,
)
'''
monitor = 'val_loss' : 검증 손실값(val_loss)을 모니터링 함.
patience = 2 : 검증 손실값이 개선되지 않으면 학습을 2번 더 진행한 후
    조기종료(EarlyStopping)
    EarlyStopping	    과적합되면 학습 자동 중단
예시)
Epoch 1: val_loss = 0.30  ✅ 좋아짐
Epoch 2: val_loss = 0.25  ✅ 좋아짐
Epoch 3: val_loss = 0.27  ❌ 더 안 좋아짐
Epoch 4: val_loss = 0.28  ❌ 또 안 좋아짐
Epoch 5: (멈춤!)          🛑 patience=2 초과
'''

score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])