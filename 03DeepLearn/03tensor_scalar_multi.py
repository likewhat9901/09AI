import tensorflow as tf
'''
**브로드캐스팅(Broadcasting)**을 이용한 **요소별 곱셈(Element-wise multiplication)**
'''

# 첫번째 텐서 정의 : 정수형 상수 텐서(1차원 텐서, 벡터, 리스트)
a = tf.constant([1,2,3], dtype=tf.int32)
# 두번째 텐서 정의 : 스칼라(0차원 텐서, 숫자)
b = tf.constant(2, dtype=tf.int32)
# 텐서 a와 b를 요소별로 곱셈 수행
x2_op = a * b # 자동으로 브로드캐스팅 => [1, 2, 3] * 2 → [2, 4, 6]
# 결과를 출력하기 위해 numpy() 메서드로 값을 가져옴
print(x2_op.numpy())

# 텐서 재정의
a = tf.constant([10,20,10], dtype=tf.int32)
x2_op = a * b # [10, 20, 10] * 2 → [20, 40, 20]
print(x2_op.numpy())
'''
스칼라	    단일 값 (0차원 텐서)
벡터	        1차원 텐서 (리스트처럼 생김)
브로드캐스팅	작은 차원의 텐서를 자동으로 확장해서 연산 수행
요소별 연산	같은 위치의 원소들끼리 연산 (덧셈, 곱셈 등)
.numpy()	텐서 값을 NumPy 배열로 변환해서 출력 가능
'''