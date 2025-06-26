import tensorflow as tf
import pandas as pd
import numpy as np
'''
TensorFlow(tf): 딥러닝 프레임워크―신경망 정의·학습·평가 담당
pandas(pd): CSV 읽기·가공에 특화된 데이터 프레임 라이브러리
NumPy(np): 수치 계산과 배열 처리 기본 패키지
'''

# BMI 데이터 로드한 후 데이터프레임으로 변환
'''ㅇ
./resData/bmi.csv 파일을 데이터프레임으로 로드.
이후 칼럼 이름 기준으로 편하게 접근 가능 (csv["height"] 등).
'''
csv = pd.read_csv("./resData/bmi.csv")

# 데이터 정규화 후 각 컬럼에 즉시 적용
'''
키·몸무게 범위를 0~1 사이로 맞추면 학습이 안정적.
여기서는 최대값 200(cm, kg 모두)을 가정하고 단순히 200으로 나눔.
'''
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 200

# label 컬럼의 체형정보를 원-핫 인코딩 형태로 변환(딕셔너리로 정의)
'''
✅ 왜 [1, 0, 0] 식으로 3차원으로 만들었는가?
- 문자형 클래스 → 숫자 벡터(원-핫 인코딩)
딥러닝 모델은 숫자로만 학습이 가능해요. "thin", "normal", "fat"처럼 문자형 범주는 직접 사용할 수 없습니다.
→ 그래서 각 클래스를 숫자 벡터로 바꾸는 방식이 필요합니다.
그중 가장 대표적인 방법이 바로 **원-핫 인코딩 (One-hot encoding)**입니다.

🔢 원-핫 인코딩이란?
클래스가 3개 있다고 가정하면:
클래스	원-핫 인코딩
"thin"	[1, 0, 0]
"normal"	[0, 1, 0]
"fat"	[0, 0, 1]

→ 각 클래스마다 자기 자리만 1, 나머지는 0
→ 이 벡터는 3차원 (클래스 수가 3이니까)

🧠 왜 이렇게 해야 할까?
딥러닝 모델 출력층이 이런 구조를 예상하고 학습하기 때문.
tf.keras.layers.Dense(3, activation='softmax')
출력 노드 3개 → 각 노드가 "thin", "normal", "fat"일 확률을 예측
예측 결과 예: [0.1, 0.85, 0.05]
정답도 같은 형식으로 있어야 비교 가능: [0, 1, 0]
✨ 이 두 벡터를 비교해서 오차(loss)를 계산하는 게 cross-entropy
'''
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
# 변환한 데이터로 새로운 컬럼 생성
'''
apply()로 행마다 사전을 조회해 새 컬럼 label_pat에 저장.

df["컬럼명"].apply(함수, axis=0(default))
→ 해당 컬럼(Series)의 각 값마다 함수를 적용해서 새로운 값 반환

Series.apply(func)	각 원소에 func(x) 적용
DataFrame.apply(func, axis=1)	각 행에 func(row) 적용
DataFrame.apply(func, axis=0)	각 열에 func(col) 적용
'''
csv["label_pat"] = csv["label"].apply(lambda x: np.array(bclass[x])) # Series.apply(func)
'''
csv	DataFrame	여러 행과 열이 있는 표 형태
csv["label"]	Series	"label" 열 하나만 추출한 1차원 데이터
'''

print(csv.head())

# 테스트 데이터 분리
test_csv = csv[15000:20000]
# 테스트용 입력 데이터(몸무게, 키)
'''
test_csv[["weight", "height"]]
: test_csv에서 "weight"와 "height" 열만 추출해서 2차원 NumPy 배열로 바꾸는 것
    => pandas의 DataFrame 슬라이싱
결과
    weight  height
0    0.50    0.80
1    0.55    0.77
2    0.43    0.73
...

np.array(...)
- 이걸 NumPy 배열로 변환.
- 결과는 shape = (5000, 2)인 2차원 배열입니다:
결과
array([[0.50, 0.80],
       [0.55, 0.77],
       [0.43, 0.73],
       ...
])
==> 딥러닝 모델 입력은 숫자 배열 (NumPy or Tensor) 형식이 필요하기 때문.
'''
test_pat = np.array(test_csv[["weight", "height"]])

# 테스트용 정답 레이블(원-핫 인코딩)
'''
🤔 list(...)를 써서 다시 리스트로 바꾸고 np.array()를 적용하는 이유
test_csv["label_pat"]는 pandas Series 객체이고,
그 안에 있는 값 하나하나가 NumPy 배열이기 때문이에요.
즉, 이건 **"배열들의 시리즈"**
=> list()로 묶어서 2차원 배열로 변환
결과
array([
  [0, 1, 0],
  [1, 0, 0],
  [0, 0, 1],
  ...
])  # shape = (5000, 3)
'''
test_ans = np.array(list(test_csv["label_pat"]))

'''
신경망 모델 정의
=> 입력층과 출력층만으로 구성된 아주 단순한 신경망

입력레이어 : 키와 몸무게 2개의 입력값을 사용
출력레이어 : 3개의 클래스(thin, normal, fat) 사용
    활성화함수는 softmax 사용
=> 
입력: 2개 (몸무게, 키)
출력: 3개 (thin, normal, fat)

model = tf.keras.Sequential([...])  * "순차적(Sequential)"이라는 뜻!
- Sequential은 레이어를 순서대로 쌓는 모델을 만드는 클래스
- [ ... ]에 나열된 레이어들이 위에서 아래로 하나씩 실행
- 입력 → 은닉층 (없음) → 출력층  (직선형 모델 구조)
'''
model = tf.keras.Sequential([
    # shape=(2,)는 입력 데이터가 2개의 값을 가진다는 뜻
    tf.keras.layers.Input(shape=(2,)),  # 입력층
    tf.keras.layers.Dense(3, activation='softmax') # 출력층
])
'''
3 → 출력 노드가 3개. 왜 3개냐면?
BMI 클래스가 3개니까요: thin, normal, fat

activation='softmax'
출력된 3개의 숫자를 확률처럼 바꿔주는 함수
예:
모델 예측값이 [1.2, 3.4, 0.9]이라면 → softmax를 거치면
[0.1, 0.85, 0.05] 같은 식으로 바뀌어요.
thin	10%
normal	85%
fat	    5%
'''

# 손실함수(크로스 엔트로피)와 옵티마이저(SGD: 경사하강법) 정의
'''
⚙️ 모델 컴파일 (compile)
모델의 학습방법을 설정하는 부분

1️⃣ optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
옵티마이저(Optimizer)
    - 모델이 얼마나 틀렸는지를 바탕으로 학습을 조정해주는 역할
SGD: Stochastic Gradient Descent (확률적 경사하강법)
    - 가장 기본적인 학습 방법
    - 가중치를 조금씩 조정해 나가면서 정답에 가까운 값을 찾도록 도와줌
learning_rate=0.01:
    - 한 번에 얼마나 조정할지 (학습 속도 조절)
    - 너무 크면 발산, 너무 작으면 느림
    0.0001	매우 조심조심 움직임. 안정적이지만 느림
    0.01	적당한 크기. 실습용에 많이 쓰임
    0.1	    매우 빠르게 움직임. 하지만 너무 빨라서 과속하다가 놓침 가능
    
2️⃣ loss='categorical_crossentropy'
손실함수(Loss function):
    - 모델의 예측이 정답에서 얼마나 틀렸는지 수치로 나타냄
categorical_crossentropy:
    - 다중 클래스 분류 문제에서 쓰는 손실 함수
    - 예측값 [0.1, 0.8, 0.1]과 정답 [0, 1, 0] 간의 차이를 계산함
    
3️⃣ metrics=['accuracy']
    - 학습 도중 어떤 평가 지표를 확인할지 지정
    - 여기선 'accuracy' → **맞은 비율(정답률)**을 보고 싶다는 뜻
'''
model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습 데이터 준비
train_pat = np.array(csv[["weight", "height"]])
train_ans = np.array(list(csv["label_pat"]))

# 학습 진행 (실제로 "머신러닝을 수행하는" 단계)
'''
epochs : 데이터셋 전체를 35번 반복 학습
batch_size : 학습 중 한번에 사용할 데이터의 크기 지정
validation_data : 테스트 데이터를 사용해 매 epochs마다 모델의 성능을 검증
verbose : 훈련 진행 상황을 시각적으로 표시
'''
history = model.fit(
    train_pat, train_ans, # 입력값과 정답값
    epochs=35,
    batch_size=100, # 즉, 전체 학습 데이터가 15000개라면, 150번 반복하여 1 epoch을 완성
    validation_data=(test_pat, test_ans), # 매 epoch마다 이 테스트 데이터를 사용해서 검증 => 성능확인용
    verbose=1  # (0 = 아무 출력 없음, 2 = 텍스트만)
)

# 최종 정확도 출력 : 손실 값과 정확도를 반환
'''
model.evaluate(...)는 입력값을 넣고 모델이 예측한 결과를 실제 정답과 비교
test_loss: 손실 값 (오차)
test_acc: 정확도 (정답률)
'''
test_loss, test_acc = model.evaluate(test_pat, test_ans)
print("정답률 =", test_acc)

'''
🧠 전체 흐름 요약
compile()	"어떻게 학습할지 설정"
fit()	"실제로 학습 진행"
evaluate()	"학습 끝난 후 성능 확인"
'''