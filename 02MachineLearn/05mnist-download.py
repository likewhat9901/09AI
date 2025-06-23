import urllib.request as req
import gzip, os, os.path
'''
urllib.request: 인터넷에서 파일 다운로드할 때 사용.
gzip: .gz로 압축된 파일을 읽고 풀기 위한 모듈.
os: 디렉토리 만들기, 경로 다루기 등 운영체제 기능.
os.path: 경로가 존재하는지 확인, 경로 붙이기 등.

os (Operating System)
: 파이썬에서 운영체제 기능을 사용할 수 있게 도와주는 표준 라이브러리
ex.폴더 만들기, 파일 경로 다루기, 환경변수 가져오기, 파일 목록 읽기 등

os.path: os 모듈 안에 있는 경로(path)를 다루는 도구 모음
    os.path.exists(path) : 해당 경로가 존재하는지 확인 (파일이든 폴더든)
    os.path.join("a", "b") : "a/b"처럼 경로를 운영체제에 맞게 이어붙임
    os.path.isdir(path) : 폴더인지 확인
    os.path.isfile(path) : 파일인지 확인
'''

# 다운로드 진행률 표시
'''
block_num	지금까지 받은 블록 수 (몇 번째 블록까지 받았는지)
block_size	블록 하나당 바이트 수 (기본값: 8192바이트 = 8KB 정도)
total_size	전체 파일의 총 바이트 수 (예: 2,000,000바이트면 약 2MB)

형식: reporthook(block_num, block_size, total_size)
'''
def progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
    print(f"다운로드 진행률: {percent:.2f}%")

# 파일을 저장할 디렉토리 지정
savepath = "./resMnist"
# 다운로드 할 github URL
baseurl = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
# 다운로드 할 파일명을 List로 정리
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"]

# 디렉토리(폴더)가 없으면 생성
if not os.path.exists(savepath): os.mkdir(savepath)
'''
os.path.exists(path) : 해당 경로가 존재하는지 확인 (파일이든 폴더든), True or False

os.mkdir(path)
    mkdir = make directory
    경로 path에 폴더(디렉토리)를 새로 만듬.
    단, 이미 있으면 에러.
    ex. os.mkdir("myfolder")
    현재 디렉토리에 myfolder라는 폴더를 만듦
'''

# 배열(리스트)로 선언한 각 파일을 다운로드
for f in files:
    # 다운로드 및 저장 경로 조립
    url = baseurl + "/" + f
    loc = savepath + "/" + f
    print("download:", url)
    # 경로에 파일이 없으면
    if not os.path.exists(loc):
        # 다운로드 진행. url로부터 다운로드 후 loc에 저장함.
        '''
        urlretrieve(url, path[, 콜백함수])
            - Python에서 인터넷에 있는 파일을 다운로드하는 데 사용하는 함수
            - urllib.request 모듈 안에 포함
            - progress가 선택인자
            - urlretrieve()는 오직 GET 요청만 가능
        <공식문서>
        urlretrieve(url, filename(=path)=None, reporthook=None, data=None)
            reporthook: 콜백 함수
            형식: reporthook(block_num, block_size, total_size)
        urlretrieve(
            url=다운로드주소,
            filename=저장경로,
            reporthook=콜백함수
        )
        '''
        req.urlretrieve(url, loc, progress)
        # 진행률을 표시하려면 3번째 인수로 progress(콜백함수명)를 추가.


# GZip 파일 압축 해제
for f in files:
    gz_file = savepath + "/" + f # savepath = "./resMnist"
    raw_file = savepath + "/" + f.replace(".gz", "") # .gz를 빈값으로 변경
    print("gzip:", f)
    # gzip.open(gz_file, "rb"): 자동으로 압축을 풀면서 읽음.
    with gzip.open(gz_file, "rb") as fp: # 바이너리 읽기모드
        # 압축이 풀린 데이터를 한 번에 전부 읽어서 변수 body에 저장
        body = fp.read() # 파일 내용 전체
        with open(raw_file, "wb") as w: # 바이너리 쓰기모드
            w.write(body) # .gz가 없는 파일로 저장

# 실행 종료
print("ok")
'''
1️⃣ 다운로드 후
파일 이름	설명
train-images-idx3-ubyte.gz	압축된 이미지 파일
train-labels-idx1-ubyte.gz	압축된 라벨 파일

2️⃣ 압축 해제 후 --> 데이터 분석에 쓰려면, 압축을 먼저 풀어야 실제 데이터를 읽을 수 있기 때문
파일 이름	설명
train-images-idx3-ubyte	압축 풀린 원본 파일
train-labels-idx1-ubyte	압축 풀린 원본 파일
'''