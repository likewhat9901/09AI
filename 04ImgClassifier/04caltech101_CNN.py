'''
Keras 라이브러리에서 신경망을 구성하는 도구들을 불러옴.
Conv2D, MaxPooling2D, Dense 등은 CNN에서 자주 쓰이는 레이어(층)들.
    Conv2D: 이미지에 필터를 적용해 특징 추출
    MaxPooling2D: 이미지 크기를 줄이며 중요한 특징만 남김
    Activation: 비선형 처리 (ReLU 같은 함수)
    Dropout: 과적합 방지를 위해 일부 뉴런을 끔
    Flatten: 다차원 → 1차원으로 펼침
    Dense: 일반적인 완전 연결층
numpy는 숫자 데이터를 다루기 위한 필수 라이브러리.
'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

# 카테고리 지정 : 분류할 대상 클래스 목록을 5개로 설정
categories = ["chairs", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories) # 출력층에서 필요함

# 이미지 크기 지정 (입력층의 기본 이미지 크기 설정)
image_w = 64
image_h = 64

# 이전에 만들었던 npz 파일 불러오기
# 이전에 np.savez() 함수로 데이터를 훈련용, 테스트용으로 분리했었음.
data = np.load("./saveFiles/caltech_5object.npz")
# 학습용 이미지, 테스트용 이미지, 정답 데이터(Y)를 각각 분리
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]

# 학습 및 테스트 데이터 정규화 (이미지 픽셀값 범위: 0~255)
# 픽셀 데이터를 0~1 사이로 변환하여 신경망 학습 안정화
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0
# 데이터 형태 출력(샘플 개수, 64, 64, 채널 수)
# (400, 64, 64, 3) → 400장의 64x64 RGB 이미지라는 뜻.
print('X_train shape', X_train.shape)

# 모델 생성(Sequential: 레이어(층)를 순차적으로 쌓는 방식)
model = Sequential()

# 입력층 : 첫번째 합성곱(Convolution) 층
'''
Conv2D(32, (3, 3)) :
    3*3 크기의 필터 32개를 적용 → 3x3 필터(커널)로 이미지를 훑어서 특징 추출. 32개의 필터 사용.
    CNN의 필터는 미리 역할이 정해진 군대가 아니라, 그냥 무작위로 뽑힌 훈련병들이고,
    훈련을 거치면서 누군가는 저격수 역할, 누군가는 정찰병 역할을 스스로 익히는 구조예요.
padding='same' : 
    출력 크기를 동일하게 유지하는 same으로 설정 (이미지 크기를 유지하면서 가장자리도 분석)
input_shape=... :
    입력데이터 형태 지정(64*64*채널수)
Activation('relu'):
    음수는 0으로 만들고 양수만 전달 (계산 효율↑)
    즉, 0 이하 차단 + 양수 통과 → 비선형적 꺾임 구조
MaxPooling2D(2, 2):
    2x2 영역에서 최댓값만 뽑아 이미지 크기를 절반으로 줄임.
    예시)
    [ 4  1 ]
    [ 3  9 ]
    → 이 중에서 가장 큰 값은 9.
Dropout(0.25):
    과적합 방지를 위해 25%의 뉴런을 무작위로 끔.
'''
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
# 활성화함수 ReLU 적용(비선형성 추가)
model.add(Activation('relu'))
# 2*2 최대 풀링 적용
model.add(MaxPooling2D(pool_size=(2, 2)))
# 과적합 방지를 위한 드롭아웃(25% 뉴런 비활성화)
model.add(Dropout(0.25))

# 은닉층1 : 두번째 합성곱 층
'''더 깊은 필터 64개로 더 복잡한 특징을 추출함.
예: 귀, 눈, 날개 같은 고급 패턴.'''
# 3x3 크기의 필터를 64개 쓰겠다는 의미.
# 즉, 한 이미지에서 64가지 다른 시선으로 특징을 추출해낸다는 뜻
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

# 은닉층2 : 세번째 합성곱 층
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 은닉층3 : 완전 연결층(Fully Connected Layer)
'''
Flatten():
    2D 이미지 데이터(예: 16x16x64)를 1D 벡터로 펼침.
Dense(512):
    512개의 뉴런으로 전부 연결.
'''
model.add(Flatten())  # 2D 특징맵을 1D 벡터로 변환
model.add(Dense(512))  # 완전연결층(512개의 뉴런)
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 출력층 : 클래스의 개수만큼 뉴런 생성
'''
Dense(nb_classes):
    클래스 개수(5개)만큼 출력 뉴런 생성.
softmax:
    각 클래스의 확률을 출력. 가장 높은 값이 예측 결과!
'''
model.add(Dense(nb_classes))
# 다중 틀래스 분류를 위한 소프트맥스 활성화 함수 지정
model.add(Activation('softmax'))

# 손실 함수, 옵티마이저, 평가 지표
model.compile(loss='binary_crossentropy',  # 손실함수 지정
              optimizer='rmsprop',  # RMSprop(빠르게 수렴하는 알고리즘) 옵티마이저 사용
              metrics=['accuracy'])  # 평가 지표로 정확도 사용

# 모델 훈련(학습 데이터 적용)
model.fit(X_train, Y_train, batch_size=32, epochs=50)
# 모델 평가
score = model.evaluate(X_test, Y_test)
# 손실 출력
print('loss=', score[0])
# 정확도 출력
print('accuracy=', f'{score[1]*100:.2f}%')

'''
batch_size, epochs는 딥러닝에서 사용하는 하이퍼 파라미터
batch_size : 배치크기로 전체 데이터를 한번에 모두 처리하는 대신 작은 덩어리(batch)로
    나누어서 처리한다. 즉 훈련에 사용하기 위한 데이터의 개수를 표현한다.
epochs : 전체 데이터를 몇 번 반복해서 훈련할지 지정한다.
    에폭의 수가 많으면 훈련 데이터에 대해 더 많이 학습하게 되지만, 
    너무 많으면 과적합이 발생할 수 있으므로 적절하게 지정해야 한다.
'''