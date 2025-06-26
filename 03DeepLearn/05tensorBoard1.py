import tensorflow as tf
import datetime
'''
단순한 곱셈 연산을 TensorFlow의 그래프 모드로 실행하고,
TensorBoard에 그 그래프를 기록하는 과정
'''

# 로그 디렉터리 설정 (현재 시간 기반 폴더 생성)
'''log_dir: TensorBoard 로그를 저장할 폴더 (시간별로 고유하게 생성)'''
log_dir = "log_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
'''TensorBoard 로그 파일을 쓰는 객체'''
writer = tf.summary.create_file_writer(log_dir)

# @tf.function을 사용하여 그래프 모드로 실행
@tf.function # TensorFlow는 내부적으로 이를 정적 그래프로 최적화함
def compute():
    a = tf.constant(20, name="a")
    b = tf.constant(30, name="b")
    return a * b # 곱셈 연산

# 그래프 실행 및 결과 출력
result = compute()  # 내부적으로 20 * 30 = 600 계산
print("계산 결과:", result.numpy())  # .numpy()를 통해 텐서 값을 출력
'''
✅ 굳이 .numpy()를 써야 하는 이유
TensorFlow의 텐서(tf.Tensor)는 "그냥 숫자"처럼 보이지만, 내부적으로는 특별한 객체예요.
그래서 우리가 보기 좋게 출력하려면 .numpy()로 실제 값을 꺼내야 해요.
예시) tf.Tensor(600, shape=(), dtype=int32)

tf.Tensor는 말하자면:
숫자를 감싼 "포장 상자" 같은 거예요
숫자처럼 쓸 수는 있지만, 우리가 보기 좋게 출력되진 않음

.numpy()를 쓰면 그 포장을 벗기고 **진짜 숫자(NumPy 값)**만 꺼내서 보여줍니다

단순 출력 (값 확인)	        ✅ 필요함	    텐서 값만 깔끔하게 보고 싶을 때
딥러닝 모델 내부 연산	        ❌ 불필요	    TensorFlow가 내부적으로 다 처리함
그래프 모드 (tf.function 안)	⚠️ 사용 불가	    이때는 Eager Execution이 꺼져 있음
'''

# TensorBoard에 그래프 기록
with writer.as_default():
    tf.summary.graph(compute.get_concrete_function().graph)
    '''
    tf.summary.graph(...)
    → 이 그래프를 TensorBoard 로그로 기록
    
    compute.get_concrete_function().graph
    → compute 함수의 구체적인 연산 그래프 객체를 가져옴
    '''

print(f"TensorBoard write ok : {log_dir}")
'''
✅ TensorBoard에서 보기
터미널에서 다음 명령어 입력:
tensorboard --logdir=log_dir
'''