from sklearn import svm, metrics
import random, re

# 붓꽃의 csv 데이터 읽기
csv = []
# csv 파일을 읽기(r) 모드로 오픈
with open('./resData/iris.csv', 'r', encoding='utf-8') as fp:
    # 파일의 내용을 한줄씩 읽기
    for line in fp:
        # 줄바꿈 제거
        line = line.strip()
        # 컴마를 기준으로 문자열 분리
        cols = line.split(',') # 문자열을 쉼표(,) 기준으로 쪼갬 → 리스트로 만듦
        '''
        람다식 : 문자열 데이터를 실수로 변환하는 기능으로 정의
            csv파일을 통해 읽어온 데이터는 문자이므로 실수로 변환하는게 필요함
            ^ : 문자열의 시작
            [0-9\.] : 숫자(0~9) 또는 마침표(.)가 한번 이상 나타나는 패턴 표현
            + : 한번 이상 반복
            $ : 문자열의 끝
            즉, 123 혹은 12.34와 같은 문자와 매칭된다.
        '''
        # 정규포현식과 매칭되면 문자열은 float()을 통해 실수로 변환한다.
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        # n이라는 문자열이 숫자처럼 생겼으면 float로 바꾸고, 아니면 그대로 둠
        cols = list(map(fn, cols)) # fn 함수를 각 원소에 적용
        # 즉, ['5.1', '3.5', ..., 'Iris-setosa']를 → [5.1, 3.5, ..., 'Iris-setosa']로 변환
        # 리스트에 변환된 데이터를 추가.
        csv.append(cols)
        # 변환된 한 줄(cols)을 전체 데이터 리스트(csv)에 추가
        # 이걸 반복해서 CSV 전체 데이터를 2차원 리스트로 만들게 됨

# 첫번째 행(헤더) 삭제. 모델 학습에서는 필요 없음.
# 첫번째 줄의 헤더 제거(컬럼명이 입력되어 있음)
del csv[0]
# 데이터 셔플, 즉 섞어준다.
random.shuffle(csv)  # 데이터를 무작위로 섞어 편향(예: 품종이 한쪽에 몰림)을 방지.
total_len = len(csv)  #	전체 행(샘플) 수를 구해 이후 분할 크기를 계산.
print("데이터개수", total_len)

# 학습데이터(100개)와 테스트데이터(50개) 분할하기(2:1 비율)
'''
훈련에 사용하지 않은 데이터를 테스트에 활용해야 학습이 제대로 되었는지
확인할 수 있다. 따라서 데이터를 이와같이 분리해야 한다.'''
train_len = int(total_len * 2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
    # 데이터와 라벨로 분리
    data = csv[i][0:4] # 0,1,2,3열
    label = csv[i][4] # 4열 (5번째 열)
    if i < train_len:
        # 훈련용 데이터 추가(100개)
        train_data.append(data)
        train_label.append(label)
    else:
        # 테스트 데이터 추가(50개)
        test_data.append(data)
        test_label.append(label)

# 데이터를 학습시키고 예측하기
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# 정답률 구하기
ac_score = metrics.accuracy_score(test_label, pre)
print(f'정답률 = {ac_score*100:.2f}%')