from PIL import Image
import os, glob
import numpy as np
# train_test_split: 학습용과 테스트용 데이터를 자동으로 나눠주는 함수
from sklearn.model_selection import train_test_split

# 분류 대상 카테고리 5개 지정
caltech_dir = "./caltech101/101_ObjectCategories" # 이미지들이 들어 있는 폴더를 지정
categories = ["chair", "camera", 'butterfly', 'elephant', "flamingo"]
nb_classes = len(categories) # 총 클래스 수는 nb_classes = 5

# 이미지 크기 지정
image_w = 64
image_h = 64
# RGB이므로 3채널의 픽셀 수 계산
pixels = image_w * image_h * 3 # 전체 픽셀 수: 64 × 64 × 3 = 12288

# 이미지 데이터와 레이블(정답)을 저장할 리스트
X = []  # 이미지들 저장
Y = []  # 라벨들 저장
# 각 카테고리마다 반복
for idx, cat in enumerate(categories):
    # 레이블은 우선 모든 클래스에 대해 0으로 설정
    label = [0 for i in range(nb_classes)]
    # 현재 카테고리의 인덱스에 해당하는 클래스에 대해 1을 설정
    '''
    label은 원-핫 인코딩:
    chair → [1, 0, 0, 0, 0]
    camera → [0, 1, 0, 0, 0]
    '''
    label[idx] = 1
    # 이미지 폴더 설정
    image_dir = caltech_dir + "/" + cat
    # jpg 이미지 가져오기
    files = glob.glob(image_dir + "/*.jpg")

    for i, f  in enumerate(files):
        # 이미지 오픈 후 RGB 모드 변환 및 크기 조정
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        # 이미지를 numpy 배열로 변환 (딥러닝 모델은 숫자 배열만 처리할 수 있기 때문)
        # 3차원 배열로 바뀜.
        data = np.asarray(img)
        # 이미지와 레이블 데이터를 리스트에 추가
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

# 리스트를 ndarray로 변환
X = np.array(X)
Y = np.array(Y)

# 학습 전용 데이터와 테스트 전용 데이터 구분
'''
기본 비율: 75% 학습 / 25% 테스트
원한다면 이렇게 비율 지정 가능:
train_test_split(X, Y, test_size=0.2, random_state=42)
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
'''
(400, 64, 64, 3)  # X_train
(100, 64, 64, 3)  # X_test
400장 이미지
각 이미지는 64×64 크기
RGB → 채널 수 3 (Red, Green, Blue)

(400, 5)          # Y_train (one-hot)
(100, 5)          # Y_test
400개 이미지의 정답 정보
각 정답은 5개의 클래스 중 하나
([1, 0, 0, 0, 0] → chair, [0, 1, 0, 0, 0] → camera 등)
'''

# npz 포맷으로 저장
np.savez("./saveFiles/caltech_5object.npz", X_train=X_train, X_test=X_test,
         Y_train=Y_train, Y_test=Y_test)
print("Task Finished..!!", len(Y))


