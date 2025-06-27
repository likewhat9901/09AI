import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.random_ops import categorical

# 학습할 이미지가 저장된 폴더 및 카테고리 설정
root_dir = "./kfood"
categories = ["FriedChicken", "Kimchi", "MiyeokGuk", "Ramen", "Samgyeopsal"]
image_size = 224 # 긴 변을 기준으로 크기 조정

X = []  # 이미지 데이터
Y = []  # 레이블 데이터

# 이미지 로드 및 리사이징 함수 (비율 유지)
def load_and_resize_image(path):
    # 이미지 경로를 통해 로드
    img = cv2.imread(path)
    # 이미지가 없는 경우
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB 변환

    # 이미지의 세로 및 가로 길이
    h, w, _ = img.shape
    # 가로와 세로 길이 중 긴쪽에 맞춰서 이미지 사이즈 조정
    if h > w:
        new_h, new_w = image_size, int(w * (image_size / h))
    else:
        new_h, new_w = int(h * (image_size / w)), image_size


    img = cv2.resize(img, (new_w, new_h))

    # 정사각형으로 패딩 추가
    pad_h = (image_size - new_h) // 2
    pad_w = (image_size - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_h, image_size - new_h - pad_h,
                             pad_w, image_size - new_w - pad_w,
                             cv2.BORDER_CONSTANT, value=[0,0,0])
    return img

# 데이터 수집
all_data = []
# 음식 카테고리(5개) 리스트를 통해 반복
for idx, category in enumerate(categories):
    # 루트폴더와 카테고리명을 합쳐서 이미지의 경로를 설정
    image_dir = os.path.join(root_dir, category)
    # 해당 경로의 모든 파일을 리스트로 반환
    files = glob.glob(image_dir + "/*")
    print(f"{category} 처리 중... ({len(files)}개)")

    # 모든 이미지를 대상으로 리사이즈 진행
    for file in files:
        img = load_and_resize_image(file)
        if img is not None:
            all_data.append((img, idx))  # (이미지 데이터, 레이블)

# 데이터 섞기
np.random.shuffle(all_data)

# 데이터 분리
X_data = np.array([x[0] for x in all_data])
Y_data = np.array([x[1] for x in all_data])

# 훈련/테스트 데이터 분리 (80:20)
X_train, X_test, Y_train, Y_test = (
    train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
)

# Numpy 배열 저장
np.savez(root_dir + "/kfood_dataset.npz", X_train=X_train, X_test=X_test,
         Y_train=Y_train, Y_test=Y_test)
print("Task Finished..!!")