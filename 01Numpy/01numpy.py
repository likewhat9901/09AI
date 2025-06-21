# NumPy 라이브러리를 np라는 별칭으로 불러오기
import numpy as np

'''
ndarray
: 넘파이 기반의 데이터 타입
다차원(Multi-dimension) 배열을 쉽게 생성하고 다양한 연산을 수행
'''

# 1차원 배열 : 3행 -> 선형벡터. 행도 열도 아님. 길이 3짜리 1차원 배열
array1 = np.array([1,2,3])
print('array1 타입1:', type(array1)) # <class 'numpy.ndarray'>
print('array1 형태1:', array1.shape) # (3,)

# 2차원 배열 : 2행 3열
array2 = np.array([[1,2,3],[4,5,6]])
print('array2 형태2:', array2.shape) # (2, 3)

# 2차원 배열 : 1행 3열
array3 = np.array([[1,2,3]])
print('array3 형태3:', array3.shape) # (1, 3)

# ndim : 차원표시
print("array1: {0}차원".format(array1.ndim))  # 1
print("array2: %d차원" % array2.ndim)         # 2
print("array3: %d차원" % array3.ndim)         # 2
