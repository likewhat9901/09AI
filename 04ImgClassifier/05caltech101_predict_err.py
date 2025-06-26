import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
import numpy as np

# 카테고리 지정
categories = ["chairs", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

# 이미지 크기 지정 (입력층의 기본 이미지 크기 설정)
image_w = 64
image_h = 64

# 이전에 만들었던 npz 파일 불러오기
data = np.load("./saveFiles/caltech_5object.npz")
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]

# 학습 및 테스트 데이터 정규화 (이미지 픽셀값 범위: 0~255)
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0
print('X_train shape', X_train.shape)

# 모델 생성(Sequential: 레이어(층)를 순차적으로 쌓는 방식)
model = Sequential()

# 입력층 : 첫번째 합성곱(Convolution) 층
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 은닉층1 : 두번째 합성곱 층
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

# 은닉층2 : 세번째 합성곱 층
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 은닉층3 : 완전 연결층(Fully Connected Layer)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 출력층 : 클래스의 개수(5개)만큼 뉴런 생성
model.add(Dense(nb_classes))
# softmax는 클래스별 확률을 출력
model.add(Activation('softmax'))

# 손실 함수, 옵티마이저, 평가 지표
model.compile(loss='binary_crossentropy',  # 손실함수 지정
              optimizer='rmsprop',  # RMSprop(빠르게 수렴하는 알고리즘) 옵티마이저 사용
              metrics=['accuracy'])  # 평가 지표로 정확도 사용

'''Keras 2.11 이후 버전부터는 save_weights()를 사용할 때
파일 이름이 반드시 **.weights.h5**로 끝나야 한다고 강제하고 있습니다.'''
# 모델 훈련 및 hdf5로 저장 (.hdf5 : 가중치만 담는 파일 포맷 - 숫자 값만 파일에 저장)
hdf5_file = "./saveFiles/caltech_5object_model.weights.h5"
if os.path.exists(hdf5_file):
    # model.load_weights(...) : 가중치 모델 불러오기
    model.load_weights(hdf5_file)
else:
    # 없으면 학습 후 가중치 모델 저장
    model.fit(X_train, Y_train, batch_size=32, epochs=50)
    # model.save_weights(...) : 훈련 직후 가중치를 파일로 저장
    model.save_weights(hdf5_file)

# 모델 평가 : 테스트 데이터를 통해 모델 예측
'''
model.predict(X_test)
    (테스트 샘플 수 × 5) 배열 -> 출력층(Dense)이 클래스 수만큼 뉴런을 만들기 때문. + softmax는 클래스별 확률을 출력
    각 행은 [p(chairs), p(camera), …, p(flamingo)] 식의 확률 벡터
'''
pre = model.predict(X_test)
# 평가된 결과 데이터를 통해 반복
for i, v in enumerate(pre):
    # 모델이 예측한 클래스를 통해 가장 확률이 높은 인덱스
    pre_ans = v.argmax()
    # 실제 정답 클래스 중 가장 확률이 높은 인덱스
    ans = Y_test[i].argmax()
    # 원본 이미지 데이터
    dat = X_test[i]
    # 예측이 정답이면 반복문의 처음으로 돌아감. 밑의 코드 실행X
    if ans == pre_ans: continue
    
    # 분류가 잘못된 경우 NG 메세지 출력
    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)
    
    # 예측에 실패한 이미지 저장 경로 설정
    fname = "./predict_error/" + str(i) + "-" + categories[pre_ans] + \
        "-ne-" + categories[ans] + ".png"
    # 0~1 범위의 이미지 데이터를 0~255로 변환
    dat *= 255
    # 넘파이 배열을 이미지 객체로 변환 후...
    img = Image.fromarray(np.uint8(dat))
    # 저장
    img.save(fname)

score = model.evaluate(X_test, Y_test)
# 손실 출력
print('loss=', score[0])
# 정확도 출력
print('accuracy=', f'{score[1]*100:.2f}%')
