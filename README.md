# live_tracking
(수정 중)

### 현재 상황
webcam -> face detection -> 디텍션 정보 얻어오는 거까지 진행하였습니다!

### 고민 상황
1) 디텍션 정보를 어떻게 담으면 좋을지.  
+ 현재는 데이터 클래스 하나에 한 좌표 값들을 넣었음(signal별로 분류)
+ 시그널들을 한 리스트로 묶었음
+ 즉 얼굴 하나의 정보 안에 [face_full, face_core_landmark, face_all_landmark] 이렇게 구성
+ 얼굴이 여러 개일 수 있으므로 최종적인 데이터 구조는 [[face_full, face_core_landmark, face_all_landmark](얼굴 1), [face_full, face_core_landmark, face_all_landmark](얼굴 2) ...] 이런 식으로 구성.  

하였는데 더 좋은 방식이 있을까요??

2) 효율적인 데이터 처리
얼굴이 여러개일 때 버블 정렬하는 방식처럼 일일이 얼굴 좌표 값들을 처리하고 있는데 더 효율적인 알고리즘이 있을까요???

### 다음 작업
object detection 모델 연결 및 디텍션 정보 전처리(1~2주 예상)
