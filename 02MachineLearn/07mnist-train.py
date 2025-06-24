from sklearn import model_selection, svm, metrics
'''
svm: SVM (Support Vector Machine) 분류기
metrics: 정확도, 리포트 등 평가 도구
model_selection: 학습/테스트 분리 등 (이번 코드에선 안 씀)
'''

# CSV 파일을 읽어 들이고 가공하기
def load_csv(fname):
    # 레이블과 이미지 데이터를 저장할 리스트
    labels = []
    images = []
    # 파일을 읽기모드로 오픈한 후 레이블과 이미지 데이터를 추가
    with open(fname, "r") as f: # as f: 파일 객체를 f라는 이름으로 사용하겠다
        # 파일의 내용을 한줄씩 읽은 후..
        for line in f: # 파일을 한 줄(line)씩 자동으로 읽어오는 반복문
            # 콤마로 구분되어 있으므로 split해서 배열로 반환
            cols = line.split(",")
            # 배열의 길이가 2미만이면 정상적인 데이터가 아니므로 통과
            if len(cols) < 2: continue
            # 0번 인덱스는 레이블이므로 삭제 후 반환되는 값을 labels 리스트에 추가하고..
            # 반환되는 값: 삭제된 데이터.
            labels.append(int(cols.pop(0)))
            # 나머지 부분은 이미지의 픽셀데이터를 가져와서 정규화(0~1사이의 값) 시켜준다.
            vals = list(map(lambda n : int(n) / 255, cols))
            '''
            이미지의 픽셀 값을 0~1 사이로 정규화하는 이유
                : 기계학습 모델이 더 잘 학습하도록 돕기 위해서
                - 스케일을 맞추기 위해	SVM 같은 알고리즘은 입력값의 크기에 민감해서,
                  값의 범위가 너무 크면 계산이 불안정하거나 느려져요.
                - 모델 수렴 속도 향상	값이 적당한 범위(0~1)에 있으면 학습이 빠르고 안정적이에요.
                - 신경망/머신러닝 모델의 기본 전처리 관행	대부분의 모델이 정규화를 가정하고 동작합니다.
            '''
            # images 리스트에 추가한다.
            images.append(vals)
    # 딕셔너리로 만들어서 반환
    return {"labels": labels, "images": images}

# 함수를 실행하여 CSV 파일을 정규화된 데이터로 변환
data = load_csv("./resMnist/train.csv")
test = load_csv("./resMnist/t10k.csv")

# 학습하기
'''
clf는 SVM(Support Vector Machine) 분류기
아직은 아무 것도 학습하지 않은 "빈 모델"
기본 설정은: 
    - 커널(kernel): RBF (비선형)
    - C=1.0, gamma='scale'
'''
clf = svm.SVC()
clf.fit(data["images"], data["labels"]) # data["images"], data["labels"]을 이용해 모델 학습
'''
data["images"]: 각 이미지를 숫자 벡터로 바꾼 것 (28x28 이미지를 784차원 벡터로 바꾸듯이)
data["labels"]: 각 이미지의 실제 정답 (예: 숫자 0~9)

clf.fit(X, y)
①	각 이미지 벡터(X)를 기준으로, 어떤 벡터는 0, 어떤 벡터는 1처럼 레이블(y)로 나눔
②	SVM은 이 데이터를 최대한 정확하게 구분할 수 있는 결정 경계(Decision Boundary) 를 찾으려고 함
③	이 경계를 정의하는 Support Vectors 들을 찾아냄
④	결과적으로, 새 데이터가 들어오면 어느 쪽에 속하는지를 예측할 수 있는 수학적 모델이 만들어짐

즉, clf.fit(data["images"], data["labels"]) 는
👉 "이 이미지 벡터는 어떤 숫자(라벨)인지"를 바탕으로
👉 SVM이 데이터를 나눌 수 있는 경계(분류 기준)를 학습하는 과정입니다.
'''

# 예측하기
predict = clf.predict(test["images"])
'''
학습을 마친 모델 clf에게 테스트 이미지들을 넣고,
**각 이미지가 어떤 숫자(0~9)**인지 예측해 달라고 요청하는 거예요.

test["images"] = [이미지1, 이미지2, 이미지3, ...]
predict = [7, 2, 1, 0, 4, 1, ...]  # 모델이 예측한 결과
'''

# 결과 확인하기
'''
test["labels"]: 실제 정답
predict: 모델이 예측한 값

test["labels"]  = [7, 2, 1, 0, 4, 9, ...]   # 진짜 정답
predict         = [7, 2, 1, 0, 4, 1, ...]   # 모델이 예측
'''
ac_score = metrics.accuracy_score(test["labels"], predict)
'''
test["labels"], predict => 정답과 예측의 인자 순서가 바뀌어도 결과는 같아요.
단, 관례적으로는 accuracy_score(정답, 예측) 순서로 씁니다.
    - 모든 Scikit-learn 평가 함수는 y_true, y_pred 순서를 기본 설계로 가정하고 있어요.
    - 다른 함수들 (precision_score, recall_score 등)은 순서 바꾸면 결과가 달라질 수도 있어요.
'''
cl_report = metrics.classification_report(test["labels"], predict)
'''
🧾 결과 확인 ②: 리포트 (정밀도, 재현율 등)
각 숫자(0~9)에 대해,
    - 모델이 얼마나 잘 맞췄는지
    - 어떤 숫자를 헷갈려했는지
    - 정확도, 정밀도, 재현율 등 다양한 지표를 보여줌
'''
print("정답률 =", ac_score)
print("리포트 =")
print(cl_report)
'''
출력예시)
              precision    recall  f1-score   support

           0       0.98      0.97      0.98       980
           1       0.99      0.98      0.98      1135
           2       0.97      0.96      0.97      1032
           ...

precision(정밀도) :	    이 숫자라고 예측한 것 중에서 진짜 맞은 비율
    - "내가 맞다고 한 것들 중에 실제로 맞은 건 몇 개?"
    - 고양이라고 예측한 것: A, B, D → 총 3개
    - 그 중 실제 고양이인 것: A, B → 2개
    - 정밀도 = 2 / 3 = 맞힌수 / 예측수 = 0.66
    => 얼마나 "깔끔하게" 맞췄나
    
recall(재현율) :	        실제 이 숫자였던 것 중에서 맞게 예측한 비율
    - "실제로 맞는 것들 중에 내가 몇 개나 맞췄어?"
    - 실제 고양이인 것: A, B, C → 총 3개
    - 그 중 내가 맞춘 것: A, B → 2개
    - 재현율 = 2 / 3 = 맞힌수 / 실제수 = 0.66
    => 얼마나 "놓치지 않고" 맞췄나
    
f1-score(F1 점수) :	    precision과 recall을 평균 낸 점수 (정밀도와 재현율의 조화 평균)
    - F1 = 2 * (precision * recall) / (precision + recall)
    - precision만 높고 recall이 낮거나, 반대인 경우 전체 평가가 왜곡될 수 있어요.
    - F1-score는 두 지표가 균형 있게 높을 때만 높은 점수를 줘요.
    - "정밀도도 중요하고, 재현율도 중요할 때" 유용한 종합 점수
    
support(지원 수, 개수) :  해당 숫자가 실제로 몇 번 나왔는지
    - 해당 클래스(숫자 등)가 실제로 테스트 데이터에서 몇 번 나왔는가
    - 즉, test["labels"] 안에 3이라는 숫자가 총 몇 번 있는가?
    - support는 해당 클래스의 중요성이나 분포를 이해하는 데 도움을 줘요.
'''