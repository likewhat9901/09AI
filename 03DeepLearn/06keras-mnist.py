from tensorflow.keras.datasets import mnist      # 손글씨 데이터
from tensorflow.keras import Sequential          # 층을 순서대로 쌓는 모델
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils               # 원-핫 인코딩 헬퍼
'''
MNIST: 28×28 흑백 손글씨 숫자(0-9) 70 000장(훈련 60 000, 테스트 10 000).
Sequential: 가장 단순한 모델 컨테이너(위→아래로 층 쌓기).
Dense: 완전 연결(FC) 층,
Dropout: 과적합 방지,
Activation: 활성화 함수.
Adam: 대표적인 최적화 알고리즘.
utils.to_categorical: 숫자 레이블 → 원-핫 벡터로 변환.
'''

# MNIST 데이터 로드
'''
X_train: 60,000개의 훈련 이미지 (28×28 픽셀)
y_train: 그에 대응하는 숫자 라벨 (정수 0~9)
X_test: 10,000개의 테스트 이미지
y_test: 테스트용 정답
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 호출 한 줄로 훈련·테스트 세트가 튜플로 반환.

# 데이터 전처리
'''
reshape(?, 784)	    28×28 픽셀을 1차원 벡터 784개 값으로 펼침(FC층 입력 형태).
astype('float32')	TF 계산 효율 위해 실수형 변환.
/=255	            픽셀값 0255 → 01 정규화 (학습 안정).
'''
'''
데이터를 float32로 변환하고 정규화한다.
28x28 크기의 2차원 이미지를 784차원의 1차원 배열로 변환하고
훈련용 6만개, 테스트용 1만개로 나눈다.
784 = 28*28

이미지는 28×28 → 784 픽셀 벡터로 펴줌 (Dense층은 1D 벡터 입력만 가능)
'''
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_train /= 255
X_test /= 255

# 원-핫 인코딩 형태로 변환
'''
정수형 레이블 데이터를 0~9까지의 카테고리를 나타내는 배열(길이 10의 벡터)로 변환한다.
즉, 원-핫 인코딩 형태로 변환한다.
예) 3 → [0 0 0 1 0 0 0 0 0 0]
    => 모델 출력 10-노드 softmax와 1:1 대응.
'''
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# 모델 구조 정의하기
'''
레이어 구성: 입력(784) → 512 → DR 0.2 → 512 → DR 0.2 → 10
Dropout 두 번으로 훈련 시 랜덤하게 노드를 끊어 일반화.
'''
# Sequential() : 모델의 각 레이어를 순차적으로 추가하는 방식의 신경망 구성
model = Sequential()
# Dense(n) : 완전연결층으로 n개의 뉴런을 가진다. 입력데이터는 784차원으로 설정
model.add(Dense(512, input_shape=(784,)))   # 512개의 노드를 가진 완전연결층
# 활성화 함수는 ReLU로 설정
model.add(Activation('relu'))               # ReLU 함수로 비선형성 부여
# 과적합 방지를 위해 20%의 뉴런을 학습에서 제외함
model.add(Dropout(0.2))                     # 20% 노드 임시 끔 (과적합 완화)
#---------------------------------------------
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#----------------------------------------------
# 여기까지 은닉층, 다음부터는 출력층으로 구성됨.
# 즉 2개의 은닉층과 1개의 출력층.
# 10개의 뉴런 및 각 클래스(숫자0~9)에 속할 확률을 계산함.
model.add(Dense(10))                        # 클래스 수 10
# 각 숫자일 확률 출력 (총합 1)
model.add(Activation('softmax'))            # 확률 분포 출력

# 모델 구축
'''
손실함수 : 크로스 엔트로피
옵티마이저 : 아담
학습 중 모델의 정확도를 측정하도록 설정
'''
model.compile(
    loss='categorical_crossentropy', # 다중 분류 문제니까 categorical_crossentropy 사용
    optimizer=Adam(),
    metrics=['accuracy']
)

# X_train / y_train을 사용해 모델 학습
'''
기본 설정: epoch=1, batch_size=32
hist에는 학습 과정의 손실과 정확도 정보가 담김
'''
hist = model.fit(X_train, y_train)

# 테스트 데이터로 평가
score = model.evaluate(X_test, y_test, verbose=1)
print('loss=', score[0]) # 오차값
print('accuracy=', score[1]) # 정답률

'''
🎯 요약 흐름
1. 데이터 준비: 로드 → 정규화 → reshape → 원-핫 인코딩
2. 모델 구성: Dense + ReLU + Dropout → softmax
3. 학습: fit()으로 학습
4. 평가: evaluate()로 정확도 확인
'''