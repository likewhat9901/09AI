import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV

# MNIST 학습 데이터와 테스트 데이터 파일 로드
train_csv = pd.read_csv("resMnist/train.csv", header=None)
test_csv = pd.read_csv("resMnist/t10k.csv", header=None)
print(train_csv)

# 필요한 열 추출
'''
첫번째 열은 라벨, 두번째부터는 특성 데이터이므로 추출하여 저장한다.'''
train_label = train_csv.iloc[:, 0]
train_data = train_csv.iloc[:, 1:577]
test_label = test_csv.iloc[:, 0]
test_data = test_csv.iloc[:, 1:577]
print("학습 데이터의 수 =", len(train_label))

# 그리드 서치 파라미터 설정
'''
svm의 다양한 매개변수 조합을 설정할 수 있다.
C : 정규화 강도 => 설정한 강도에 따라서 얼만큼 완벽한 분리를 할 수 있을지 결정
kernel : 커널함수의 종류 => 데이터를 어떻게 분류할 것인가 결정(linear, rbf, poly 등)
gamma : RBF 커널의 감마 값 => 데이터의 패턴을 설정할 수 있음
'''
params = [
    # 선형 커널의 경우, C의 값만 조정. (linear 커널은 gamma 필요 없음)
    {"C": [1,10,100,1000], "kernel":["linear"]},
    # RBF(Radial Basis Function) 커널의 경우, C와 gamma 값을 조정.
    # RBF는 커널이 곡선을 그리므로 gamma가 중요
    {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
]
'''
SVM은 커널 종류마다 의미 있는 하이퍼파라미터가 다릅니다
    linear	    C
    rbf 	    C, gamma
    poly	    C, degree, coef0 등
'''
'''
🔑 C란?
: C는 규제(regularization) 강도를 조절하는 하이퍼파라미터
 => 모델이 학습 데이터의 오류를 얼마나 허용할지 결정

📈 C 값이 클수록?
- 모델이 오차를 허용하지 않으려 함
- 학습 데이터를 최대한 정확하게 분류하려 함
- 하지만 너무 크면 → 과적합(overfitting) 위험

📉 C 값이 작을수록?
- 약간의 오차를 허용
- 대신 더 단순하고 일반화된 모델을 만들려고 함
- 너무 작으면 → 과소적합(underfitting) 위험

💡 [1, 10, 100, 1000] 이 의미하는 것
: 여러 개의 C 값을 시도해 보면서 어떤 값이 성능이 제일 좋은지 찾기 위해 하이퍼파라미터 튜닝을 하는 것
1	    규제가 약간 강함 (오차 좀 허용)
10	    중간
100	    규제 약함 (오차 거의 허용 안 함)
1000	오차 허용 거의 없음 (매우 복잡한 모델 가능)
'''


# 그리드 서치 수행 : 설정된 파라미터 조합 중 최적의 값을 찾음
clf = GridSearchCV(svm.SVC(), params, n_jobs=-1)
'''
GridSearchCV(...)	모든 조합을 다 테스트해주는 그리드 서치 도구
svm.SVC()	        SVM(Support Vector Classifier) 모델 객체 생성
params	            실험할 하이퍼파라미터 조합 딕셔너리
    => 즉, params로 정한 여러 조합의 C, kernel, gamma를 시도한다
n_jobs=-1	        CPU 코어를 최대한 사용해서 빠르게 처리함 ✅
    "얼마나 많은 CPU 코어를 사용할까?"를 설정하는 옵션
    n_jobs=1	하나의 코어만 사용 (기본값)
    n_jobs=2	CPU 2개만 사용
    n_jobs=-1	가능한 모든 CPU 사용 (💨 병렬처리로 가장 빠름!)
    
+ cv=5(교차검증 횟수) 같은 옵션도 추가가능
'''

# 내부적으로는 여러 C, kernel, gamma 조합을 시도해서, **가장 성능 좋은 모델(best estimator)**을 찾음
clf.fit(train_data, train_label) # train_data, train_label을 이용해 모델 학습

# 학습 데이터로 최적의 모델을 학습
print('학습기 =', clf.best_estimator_)
# 예: SVC(C=100, kernel='rbf', gamma=0.001) 이런 식으로 나옴
print('---------------------------')
'''
train_data와 test_data의 컬럼이 일치하는지 확인
컬럼 순서/이름이 다르면 오류나거나 예측 정확도 떨어질 수 있음
'''
print(train_data.columns)
print(test_data.columns)
print('---------------------------')

# 테스트 데이터 확인(예측 수행)
pre = clf.predict(test_data) # test_data를 넣어서 **예측 결과(pre)**를 얻음
# 예측값과 실제 레이블 비교하여 정확도 계산
ac_score = metrics.accuracy_score(pre, test_label)
'''
metrics.accuracy_score(pre, test_label)
리스트(또는 배열) 형태로 된 pre와 test_label을 하나씩 순서대로 짝지어 비교해서 정확도를 계산

✅ 형태 예시:
pre = [1, 0, 1, 1, 0]         # 예측 결과
test_label = [1, 0, 0, 1, 0]  # 실제 정답
정확도 = 4 / 5 = 0.8 (80%)

- pre, test_label은 리스트나 넘파이 배열 형식이어야 함
- 각 요소를 순서대로 하나씩 비교해서 정확도 계산
- 딕셔너리(dict)는 사용할 수 없음
'''
print("정답률 =", ac_score)