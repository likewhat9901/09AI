import numpy as np

# 정수형 2차원 배열 생성
aArray = np.array([[1,2,3], [4,5,6]])
print('aArray\n', aArray)
'''
 [[1 2 3]
 [4 5 6]]
'''

# 실수형 2차원 배열 생성. 전체가 1. 으로 초기화된다.
bArray = np.ones((2,3))
print('bArray\n', bArray)
'''
 [[1. 1. 1.]
 [1. 1. 1.]]
'''

# 배열연결1 : r_ 즉 Row(행)을 기준으로 연결하므로 4행 3열이 된다.
R1 = np.r_[aArray, bArray] # 아래로 연결
print('결과1\n', R1) # 전부 실수로 바뀜 (자동 형변환)
'''
 [[1. 2. 3.]
 [4. 5. 6.]
 [1. 1. 1.]
 [1. 1. 1.]]
'''

# 배열연결2 : c_ 즉 Column(열)이 기준이므로 2행 6열이 된다.
R2 = np.c_[aArray, bArray] # 오른쪽으로 연결
print('결과2\n', R2)
'''
 [[1. 2. 3. 1. 1. 1.]
 [4. 5. 6. 1. 1. 1.]]
'''

# 형태가 같은 ndarray 끼리는 사칙연산을 수행할 수 있다.
R3 = aArray + bArray
print('결과3\n', R3)
'''
 [[2. 3. 4.]
 [5. 6. 7.]]
'''

R4 = aArray - bArray
print('결과4\n', R4)
#  [[0. 1. 2.]
#  [3. 4. 5.]]

R5 = aArray * bArray
print('결과5\n', R5)
#  [[1. 2. 3.]
#  [4. 5. 6.]]

R6 = aArray / bArray
print('결과6\n', R6)
#  [[1. 2. 3.]
#  [4. 5. 6.]]

# 행과 열에 하나의 값을 전체적으로 적용할 수 있다.
R7 = aArray + 2
print('결과7\n', R7)
#  [[3 4 5]
#  [6 7 8]]

R8 = aArray * 2
print('결과8\n', R8)
#  [[ 2  4  6]
#  [ 8 10 12]]

'''
브로드캐스팅(Broadcasting)
NumPy의 핵심 기능 중 하나로, 서로 다른 모양(shape)의 배열끼리도 
연산이 가능하도록 자동으로 크기를 맞춰주는 기능.
'''
'''
배열끼리의 사칙연산에서는 배열의 크기가 완전히 일치하지 않아도
한쪽 차원의 길이가 1 혹은 0이면 자동으로 크기를 확장한 후 계산한다.
'''
C = np.array([[1,2,3]])
print('aArray\n', aArray)
print('aArray형태:', aArray.shape) # 2행 3열 / (2, 3)
print('C', C)
print('C의 형태', C.shape) # 1행 3열 / (1, 3)

'''
크기가 서로다른 2개의 배열을 연산하면 Broadcasting(브로드캐스팅) 규칙이 적용된다.
아래의 경우 배열 C가 2행 3열로 확장되어 연산된다.
=> C = np.array([[1,2,3]]) => C = np.array([[1,2,3], [1,2,3])
'''
R9 = aArray + C
print('결과9\n', R9)
#  [[2 4 6]
#  [5 7 9]]