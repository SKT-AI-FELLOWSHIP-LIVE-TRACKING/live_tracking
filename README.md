# live_tracking
(수정 중)

### 현재 상황
+ FastMOT 구현 완료
+ 현재 detection -> tracking -> tracking -> tracking -> tracking
+ 4개의 프레임 skip 중
+ tracking으로 detection을 놓치는 경우 detection 프레임으로 건너뛰기

### To Do
+ object detection 모델 변경
+ (가로 고정) target_height 기준 구현
