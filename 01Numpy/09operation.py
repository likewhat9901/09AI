import numpy as np

# 정수형 2차원 배열 생성
aArray = np.array([[1,2,3], [4,5,6]])
print('aArray\n', aArray)

# 실수형 2차원 배열 생성. 전체가 1. 으로 초기화된다.
bArray = np.ones((2,3))
print('bArray\n', bArray)

# 배열연결1 : r_ 즉 Row(행)을 기준으로 연결하므로 4행 3열이 된다.
R1 = np.r_[aArray, bArray]
print('결과1\n', R1)

# 배열연결2 : c_ 즉 Column(열)이 기준이므로 2행 6열이 된다.
R2 = np.c_[aArray, bArray]
print('결과2\n', R2)

# 형태가 같은 ndarray 끼리는 사칙연산을 수행할 수 있다.
R3 = aArray + bArray
print('결과3\n', R3)

R4 = aArray - bArray
print('결과4\n', R4)

R5 = aArray * bArray
print('결과5\n', R5)

R6 = aArray / bArray
print('결과6\n', R6)

# 행과 열에 하나의 값을 전체적으로 적용할 수 있다.
R7 = aArray + 2
print('결과7\n', R7)

R8 = aArray * 2
print('결과6\n', R8)

'''
배열끼리의 사칙연산에서는 배열의 크기가 완전히 일치하지 않아도
한쪽 차원의 길이가 1 혹은 0이면 자동으로 크기를 확장한 후 계산한다.
'''
C = np.array([[1,2,3]])
print('aArray형태:', aArray.shape) # 2행 3열
print('C의 형태', C.shape) # 1행 3열

'''
크기가 서로다른 2개의 배열을 연산하면 Broadcasting(브로드캐스팅) 규칙이 적용된다.
아래의 경우 배열 C가 2행 3열로 확장되어 연산된다.
=> C = np.array([[1,2,3]]) => C = np.array([[1,2,3], [1,2,3])
'''
R9 = aArray + C
print('결과9\n', R9)