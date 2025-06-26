import tensorflow as tf

# 상수 텐서 정의
'''
name: TensorFlow 내부 이름
- TensorFlow 내부에서 계산 그래프를 그릴 때 붙이는 레이블이에요.
- 이건 TensorFlow의 내부 로깅, 시각화, 변수 관리 등에만 영향을 줍니다.
'''
a = tf.constant(120, name="a")
b = tf.constant(130, name="b")
c = tf.constant(140, name="c")

# tf.Variable 정의. 값이 변경 가능한 텐서 정의
# 초기값이 0이고 이름이 "v"인 변수 텐서
v = tf.Variable(0, name="v")

# 합 계산 후 변수에 할당
calc_op = a + b + c
'''tf.Variable을 유지한 상태로 값을 바꿀때는 .assign() 사용해야 함.'''
v.assign(calc_op) # v.assign(...)을 통해 변수 v에 결과값을 직접 할당

# 값 출력
print(v.numpy())
