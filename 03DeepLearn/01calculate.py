# TensorFlow 임포트
import tensorflow as tf

# 상수 텐서 정의(그냥 숫자 1234를 변하지 않는 텐서 형태로 만든 것)
'''
텐서는 TensorFlow의 기본 데이터 단위
=> 스칼라(숫자 하나), 벡터(리스트), 행렬(2차원 배열) 등 다양한 형태를 담을 수 있음.

tf.constant(5)                    # 스칼라 텐서 (0차원)
tf.constant([1, 2, 3])            # 벡터 텐서 (1차원)
tf.constant([[1, 2], [3, 4]])     # 행렬 텐서 (2차원)

반대로 학습 가능한 변수는 tf.Variable()
'''
a = tf.constant(1234)
b = tf.constant(5678)

# 함수 정의
@tf.function # 텐서플로의 그래프 모드 최적화를 적용하는 데코레이터
def add_op(a,b):
    return a+b

# 함수 호출 후 결과를 Numpy 배열로 변환하여 출력
'''
NumPy 배열 = 숫자 계산에 특화된 리스트

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)  # → [5 7 9] (요소별 덧셈)

일반 리스트
[1, 2, 3] + [4, 5, 6]  # → [1, 2, 3, 4, 5, 6] (붙이기)
'''
res = add_op(a, b).numpy()
print(res)