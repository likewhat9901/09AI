import numpy as np

'''
ndarray에 저장하는 데이터는 숫자, 문자열, boolean등 모두 가능하다.
단 연산의 특성상 같은 자료형이어야 한다.
'''

# 정수만으로 리스트 생성
list1 = [1,2,3]
# 리스트를 ndarray로 변환
# 리스트를 넘파이 배열로 변환하면 모든 요소가 동일한 타입으로 저장
array1 = np.array(list1)

# 정수형 배열로 자동변환
print('array1 출력:', array1) # [1 2 3]
# 배열의 타입확인 : ndarray
print('array1 타입:', type(array1))
# 배열에 저장된 데이터 타입 확인 : int32
print('array1 dtype:', array1.dtype) # int64

# 실수 포함 리스트 생성
list2 = [1,2,3.0]
# 배열로 변환 시 모든 데이터가 실수로 변환.
array2 = np.array(list2)

# 실수형 배열로 자동변환
print('array2 출력:', array2) # [1. 2. 3.]
print('array2 타입:', type(array2)) # <class 'numpy.ndarray'>
# 실수가 포함되었으므로 float64로 출력된다.
print('array2 dtype:', array2.dtype)

# 문자열이 포함된 리스트 생성
list3 = [1,2,'test']
array3 = np.array(list3)

# 모든 데이터가 문자열로 변환되어 출력된다.
print('array3 출력:', array3) # ['1' '2' 'test']
print('array3 타입:', type(array3)) # <U21 (최대 글자 수는 21자)
print('array3 dtype:', array3.dtype)

'''
astype()으로 명시적 형변환
'''
# astype() : ndarray의 데이터 타입을 변경하는 함수
# 정수형을 실수형으로 변경
'''
int → float64 변환
'''
array_int1 = np.array([1,2,3])
array_float1 = array_int1.astype('float64') # int → float64 변환
print(array_float1, array_float1.dtype)

# 실수형을 정수형으로 변경
'''
float → int32 변환
소수점은 버림 처리됨 (int()처럼)
'''
array_float2 = np.array([1.1,2.1,3.1])
array_int2 = array_float2.astype('int32')
print(array_int2, array_int2.dtype)

'''
(자동 형변환 우선순위)
int → float → str
→ 더 정밀한(혹은 복잡한) 타입으로 자동 업캐스팅됨
'''
'''
ndarray를 문자열(str) 배열로 변환
astype(str) 또는 astype('U')
arr.astype(str)	<U숫자>	파이썬 기본 문자열 타입
arr.astype('U')	<U숫자>	NumPy 유니코드 문자열
arr.astype('S')	<S숫자>	바이트 문자열 (ASCII)
'''