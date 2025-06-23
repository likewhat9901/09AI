import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
'''
os	경로 조작 (디렉토리, 파일 경로 만들기 등)
glob	특정 패턴에 맞는 파일을 리스트로 찾기
    glob는 “큰 덩어리, 덩어리”라는 뜻입니다.
    컴퓨터 용어로는, **“파일 이름 패턴을 이용해 여러 파일을 한꺼번에 묶어 처리한다”**는 의미로 쓰입니다.
PIL.Image	이미지를 열고 다루기 (.pgm, .jpg, .png 등)
    PIL (Python Imaging Library)
    : 파이썬에서 이미지를 열고, 수정하고, 저장할 수 있게 해주는 이미지 처리 라이브러리
matplotlib.pyplot	이미지를 시각화해서 보여주기
'''

# PGM 파일들이 저장된 디렉토리 경로
directory_path = './resMnist'

# 패턴에 맞는 모든 파일 찾기
'''
os.path.join("a", "b") : "a/b"처럼 경로를 운영체제에 맞게 이어붙임
t10k-*-*.pgm : 와일드카드(*)를 포함한 파일명
    ex)
    t10k-001-0001.pgm
    t10k-005-0130.pgm
'''
file_pattern = os.path.join(directory_path, 't10k-*-*.pgm')
pgm_files = glob.glob(file_pattern) # 해당 패턴에 맞는 파일들을 전부 찾음
print(pgm_files)

# 각 파일에 대해 한 장씩 이미지 출력
for pgm_file in pgm_files:
    img = Image.open(pgm_file) # 이미지 파일 열기

    # 이미지를 출력
    '''
    plt.imshow() : 이미지를 화면에 보여주는 함수
    cmap='gray'는 흑백 컬러맵을 지정 (PGM은 원래 흑백 이미지)
    '''
    # plt.figure() # 새창 생성
    plt.imshow(img, cmap='gray')
    plt.title(f"Image: {pgm_file}") # 파일명 제목으로 표시
    plt.axis('off') # 축 숨기기
    plt.show() # 한 장씩 표시
# plt.show() # 한 장씩 표시