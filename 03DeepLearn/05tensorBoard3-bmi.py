import tensorflow as tf
import pandas as pd
import numpy as np

# BMI 데이터 로드한 후 데이터프레임으로 변환
csv = pd.read_csv("./resData/bmi.csv")

# 데이터 정규화 후 각 컬럼에 즉시 적용
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 200

# label 컬럼의 체형정보를 원-핫 인코딩 형태로 변환(딕셔너리로 정의)
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
# 변환한 데이터로 새로운 컬럼 생성
csv["label_pat"] = csv["label"].apply(lambda x: np.array(bclass[x])) # Series.apply(func)

print(csv.head())

# 테스트 데이터 분리
test_csv = csv[15000:20000]
# 테스트용 입력 데이터(몸무게, 키)
test_pat = np.array(test_csv[["weight", "height"]])
# 테스트용 정답 레이블(원-핫 인코딩)
test_ans = np.array(list(test_csv["label_pat"]))

# 신경망 모델 정의
'''
입력레이어 : 키와 몸무게 2개의 입력값을 사용
출력레이어 : 3개의 클래스(thin, normal, fat) 사용
    활성화함수는 softmax 사용
'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 손실함수(크로스 엔트로피)와 옵티마이저(SGD: 경사하강법) 정의
model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습 데이터 준비
train_pat = np.array(csv[["weight", "height"]])
train_ans = np.array(list(csv["label_pat"]))

import datetime
# TensorBoard 로그 저장 경로 설정
log_dir = "log_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
'''
✅ 콜백(callback)이란?
딥러닝 모델이 학습하는 도중 특정 시점마다 자동으로 호출되는 함수/기능
예:
- 매 epoch이 끝날 때
- 일정 조건을 만족했을 때
- 중간 저장, 로그 기록, 학습 중단 등을 자동으로 처리

📦 TensorFlow 콜백 종류
ModelCheckpoint	    모델을 파일로 저장
EarlyStopping	    과적합되면 학습 자동 중단
TensorBoard ✅	    학습 로그를 기록해서 시각화 가능 => 학습 중간중간에 loss, accuracy 등 기록을 남겨줌
ReduceLROnPlateau	성능이 안 좋아지면 학습률 자동 조절

callbacks=[...]	fit() 함수에서 등록해서 실행됨
'''
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
'''
log_dir=...
    로그를 저장할 폴더 위치
histogram_freq=1
    매 1 epoch마다 각 레이어의 가중치 변화, 히스토그램 등도 기록함
    → 이걸 0으로 하면 그런 정보는 안 남깁니다 (속도는 빨라짐)
'''

# 학습 진행
history = model.fit(
    train_pat, train_ans,
    epochs=35,
    batch_size=100,
    validation_data=(test_pat, test_ans), # 테스트 데이터 검증
    verbose=1,
    callbacks=[tensorboard_callback], # TensorBoard 콜백 추가
)

# 최종 정확도 출력 : 손실 값과 정확도를 반환
test_loss, test_acc = model.evaluate(test_pat, test_ans)
print("정답률 =", test_acc)

# TensorBoard에 테스트 결과 기록
with tf.summary.create_file_writer(log_dir).as_default() :
    tf.summary.scalar("Test Accuracy", test_acc, step=0)
    tf.summary.scalar("Test Loss", test_loss, step=0)
'''
tf.summary.scalar(name, value, step)
→ "숫자 1개(scalar value)를 TensorBoard 로그에 기록하는 함수"
name	TensorBoard에 보여질 지표 이름 (예: "Test Accuracy")
value	기록할 실제 숫자 값 (예: 0.94)
step	시간 축으로 몇 번째 step인지 (보통 epoch 번호)
'''

print(f"TensorBoard write ok : {log_dir}")