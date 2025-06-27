from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils
import numpy as np

# root_dir = "./download"
categories = ["Gyudon", "Ramen", "Sushi", "Okonomiyaki", "Karaage"]  # (폴더 이름과 동일해야 함)
nb_classes = len(categories)
image_size = 100

# 데이터 로드 및 전처리 수행
def main():
    # Numpy 데이터 불러오기
    data = np.load("./saveFiles/japanese_food_aug.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    Y_train = data["Y_train"]
    Y_test = data["Y_test"]

    # 학습 및 테스트 데이터 정규화 (이미지 픽셀값 범위: 0~255)
    X_train = X_train.astype("float") / 255.0
    X_test = X_test.astype("float") / 255.0
    # 레이블 데이터를 원-핫 인코딩으로 변환
    Y_train = utils.to_categorical(Y_train, nb_classes)
    Y_test = utils.to_categorical(Y_test, nb_classes)
    # 모델 훈련하고 평가하기
    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

# CNN 모델 구축
def build_model(in_shape):
    # 모델 생성
    # 모델 생성
    model = Sequential()
    # 입력층 : 첫번째 합성곱(Convolution) 층
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 은닉층1 : 두번째 합성곱 층
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # 은닉층2 : 세번째 합성곱 층
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
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
    # 모델 컴파일 : 손실함수, 옵티마이저, 평가지표
    model.compile(loss='binary_crossentropy',  # 손실함수 지정
                  optimizer='rmsprop',  # RMSprop(빠르게 수렴하는 알고리즘) 옵티마이저 사용
                  metrics=['accuracy'])  # 평가 지표로 정확도 사용
    return model

# 모델 훈련
def model_train(X, Y):
    # 입력 데이터의 shape을 기반으로 모델 생성
    model = build_model(X.shape[1:])
    # 모델 학습 수행
    model.fit(X, Y, batch_size=32, epochs=1)
    # 모델 가중치 저장하기
    hdf5_file = "./saveFiles/japanese_food_aug_model.hdf5"
    model.save_weights(hdf5_file)
    return model

# 테스트 데이터로 모델 평가
def model_eval(model, X, Y):
    score = model.evaluate(X, Y)
    # 손실값, 정확도 출력
    print('loss=', score[0])
    print('accuracy=', f'{score[1] * 100:.2f}%')

if __name__ == "__main__":
    main()
