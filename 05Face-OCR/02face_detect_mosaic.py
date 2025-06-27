import cv2, sys, re

# 입력 파일 지정 (아나콘다 프롬프트에서 실행해야 함)
# CMD > python 예제파일.py 이미지경로.jpg
if len(sys.argv) <= 1:
    print("no input file")
    quit()
# 명령행을 통해 전달된 이미지 경로를 얻어와서 변수에 저장
image_file = sys.argv[1]

# 출력 파일 이름. 정규표현식을 통해 지정된 확장자를 찾은 후 'mosaic' 문자열을
# 변경한다.
output_file = re.sub(r'\.jpg|jpeg|PNG$', '-mosaic1.jpg', image_file)
print("output", output_file)

# 모자이크 강도. 숫자가 클수록 강한 모자이크 효과가 적용
mosaic_rate = 30

# 캐스케이드 파일 경로 지정
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
# 이미지를 Numpy 배열로 전환
image = cv2.imread(image_file)
# 그레이스케일
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 얼굴인식 실행하기
cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(100, 100)
)
# 얼굴이 감지되지 않으면 프로그램 종료
if len(face_list) == 0:
    print("얼굴을 인식할 수 없습니다.")
    quit()

# 얼굴이 감지되면 좌상단의 좌표와 가로, 세로 길이를 리스트로 반환
print(face_list)

# 확인한 부분을 모자이크 처리
color = (0,0,255)
for (x,y,w,h) in face_list:
    # 얼굴 부분 자르기
    face_img = image[y:y+h, x:x+w]
    # 자른 이미지를 지정한 배율로 확대/축소하기
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    # 확대/축소한 그림을 원래 크기로 돌리기
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
    # 원래 이미지에 붙이기
    image[y:y+h, x:x+w] = face_img

# 렌더링 결과를 파일에 출력
cv2.imwrite(output_file.replace("resData", "saveFiles"), image)