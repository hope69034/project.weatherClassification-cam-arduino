import cv2

# 웹캠 캡처 객체 생성
capture = cv2.VideoCapture(0)  # 웹캠 인덱스: 0 (기본 웹캠)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 저장할 비디오 파일명
output_filename = './test/output_video.avi'

# 비디오 저장을 위한 VideoWriter 객체 생성
output = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))  # 적절한 해상도와 FPS 설정
while True:
    # 비디오 프레임 읽기
    ret, frame = capture.read()

    # 캡처가 정상적으로 되었는지 확인
    if not ret:
        break

    # 프레임을 출력 비디오에 기록
    output.write(frame)

    # 화면에 프레임 출력
    cv2.imshow('Video', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용한 객체들 해제
capture.release()
output.release()
cv2.destroyAllWindows()
