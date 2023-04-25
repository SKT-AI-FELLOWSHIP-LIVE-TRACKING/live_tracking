# SKT AI fellowship 4기

## 연구 배경

최근 유튜브, 트위치 등 비디오 스트리밍 분야는 미디어 환경에서 매우 빠르게 성장하고 있습니다. 특히 라이브 스트리밍 분야 상위 5개 앱의 지난 3년 연평균 성장률이 25%를 기록하면서 사진 및 비디오 앱의 연평균 성장률 15%를 뛰어넘었습니다. 

이처럼 라이브 콘텐츠를 제작할 때, 다양한 제작 소스를 자동화하여 제공할 수 있다면 콘텐츠 제작자들에게 매우 유용한 서비스가 될 것입니다.

저희의 연구와 비슷한 주제로는 mediapipe의 Autoflip[1]과 apple의 Center Stage[2]가 있습니다.

- Autoflip

Autoflip은 자동으로 비디오를 리프레임해주는 시스템입니다. 프레임을 원하는 비율로 리프레임하여 다양한 모바일 기기들에서도 콘텐츠를 이용할 수 있도록 합니다. 하지만 Autoflip은 실시간 영상에서는 사용하지 못한다는 점에서 저희 연구와의 차이점을 가집니다.

- Center Stage

Center Stage는 아이패드에서 영상 통화를 할 때 사람의 얼굴을 인식하여 자동으로 추적하는 시스템입니다. Center Stage는 하드웨어로 추적 시스템이 조작되는 반면 저희 연구는 소포트웨어로 조작되는 점이 큰 차이점입니다. 또한 저희 연구에서는 얼굴 뿐만 아니라 사람 자체를 인식한다는 점이 다르며, 추가적으로 스포츠 경기에서 공을 추적하는 기능도 구현 중에 있습니다.

## 초기 접근 방법

저희는 라이브 트래킹 시스템을 (1) tracking할 피사체를 인식, (2) 피사체를 인식한 정보를 바탕으로 프레임 리사이징, (3) 웹앱으로 리프레임한 영상 실시간으로 송출하는 것으로 구성하였습니다. 

이 중 저는 1번과 2번에 대해 연구하였습니다.

### (1)-1 object detection

우선 원하는 피사체를 자동으로 따라가는 시스템을 만들기 위해서는 해당 피사체를 인식하는 과정이 필요합니다. 따라서 object detection 기술을 활용하였습니다. 

object detection은 영상이나 이미지에서 사람, 동물 등의 유의미한 객체를 찾기 위한 컴퓨터비전 기술입니다. object detection은 이미지 내 특정 사물을 분류하는 task인 image classification과는 다르게 탐지된 객체의 종류를 찾고(classification), 해당 객체의 위치(localization)를 사각형의 형태인 bounding box를 활용하여 찾게 됩니다.


<img width="625" alt="스크린샷 2022-10-04 오후 9 31 11" src="https://user-images.githubusercontent.com/81224613/234164996-8545b157-155a-4ee8-94a0-2d8ff8ae273e.png">

 

object detection에 주로 사용되는 모델인 yolo와 SSD 중에서 비교적 속도가 빠른 SSD 모델을 사용하여 객체를 탐지하는 데에 활용하였습니다.

### SSD

SSD[3] 모델은 객체 검출 및 분류와 bounding box를 구하는 Region Proposal이 한 번에 이루어지는 one-stage 모델입니다. 따라서 이전까지의 two-stage 모델과는 fps, 즉 연산 속도가 더 빠르다는 장점을 지니고 있습니다. 실제로 대표적인 two-stage 모델인 Faster R-CNN은 7 fps인 반면에 SSD 모델은 59 fps의 속도를 지니고 있습니다.

<img width="1101" alt="스크린샷 2022-10-06 오전 1 04 09" src="https://user-images.githubusercontent.com/81224613/234163510-eda6cf83-16bb-4aba-8e2c-d89a459edc89.png">


SSD 모델은 여러 개의 feature map에서 object detection을 수행시키는 점이 다른 object detection 모델들과의 차이점입니다. 여러 개의 feature map은 각각 다른 사이즈로 구성되어 있는데 큰 사이즈(높은 해상도)의 feature map에서는 크기가 작은 객체를 잘 인식하고, 작은 사이즈(낮은 해상도)의 feature map에서는 크기가 큰 객체를 잘 인식합니다. 다수의 feature map에서 얻은 bounding box 정보를 NMS 처리를 하면서 최종적으로 객체의 종류와 객체의 위치를 파악할 수 있습니다.

### (1)-2 face detection

<img width="170" alt="스크린샷 2022-10-06 오전 1 24 49" src="https://user-images.githubusercontent.com/81224613/234163508-1090a0c7-e2a8-46ab-8471-06836386b079.png">

얼굴 인식은 mediapipe에서 제공하는 face detection api를 사용하였습니다. mediapipe의 face detection은 BlazeFace[4]라는 경량화 face detection 모델을 통해 이루어집니다. 해당 api를 통해 이미지 내에서의 다중 얼굴을 찾고, 각 얼굴의 6개의 랜드마크(양쪽 눈, 코, 입, 양쪽 귀)들을 찾을 수 있습니다.

### (2) reframing

다음으로는 앞에서 얻은 피사체의 위치 정보를 가지고 사용자가 원하는 비율에 맞춰서 프레임을 잘라내야 합니다. 리프레임 단계에서는 피사체의 정보를 기반으로 자를 프레임의 중점을 구하고, 중점을 기준으로 임의의 비율에 맞춰서 잘라내도록 하였습니다.

**(2)-1 리프레임 최적인 중점 구하기**

피사체가 하나인 경우에는 인식된 피사체의 중점이 자를 프레임의 중점이 되기 때문에 리프레임할 중점을 구하는 것이 단순합니다. 하지만 문제는 피사체가 여러 개일 경우에 발생하였습니다. 피사체가 여러 개일 경우 상황에 따라서는 최대한 많은 피사체를 자를 프레임에 담아내거나 혹은 과감히 몇몇 피사체는 버려야 했습니다. 따라서 가장 신뢰할 수 있는 피사체(mAP score가 가장 큰 피사체)를 중심으로 리프레임을 하였습니다. 또한 해당 피사체 근처에 있는 다른 피사체들이 존재하고, 피사체 사이의 거리가 멀지 않다고 판단될 때 이들을 리프레임 시 같이 담아내려고 하였습니다. 최적 중점 구하기 관련 알고리즘 설명은 아래에서 더 자세히 다루겠습니다.

Face / Object detection으로 인식한 정보 중에서 mAP 스코어가 가장 큰 피사체를 list에 저장합니다. 이후 임의의 경계 범위(target width / height의 절반으로 설정) 안에 다른 detection의 중심 좌표가 들어올 수 있는 경우에 해당 정보를 list에 저장하고 경계 범위를 update합니다. 앞의 과정을 모두 마친 후 list에 저장된 detection 중심 좌표들의 평균을 구하면 리프레임에 최적인 중점을 얻게 됩니다.

<img width="679" alt="스크린샷 2022-10-04 오후 11 25 16" src="https://user-images.githubusercontent.com/81224613/234163503-9e5521bc-1343-4ea9-a3a3-71a407137fec.png">

위의 사진으로 예시를 들겠습니다. 각 피사체의 id는 mAP score가 높은 순서대로 부여하였고, 왼쪽에 파란 막대는 임의의 경계 범위입니다. 1번을 기준으로 경계 범위 안에 2번, 3번, 4번 피사체가 들어올 수 있는지 차례대로 비교합니다. 만약에 위의 상황에서 2번 피사체와 1번 피사체와의 거리가 설정한 경계 범위 보다 작다고 가정하면 2번 피사체도 자를 프레임에 담아내도록 하고(2번 피사체의 중점 좌표를 list에 저장) 경계 범위를 2번 피사체와 1번 피사체의 거리만큼 빼주면서 update하는 것입니다. 3번, 4번 피사체는 1, 2번 피사체와의 거리가 update된 경계 범위 보다 크다고 가정하겠습니다.

<img width="680" alt="스크린샷 2022-10-04 오후 11 32 31" src="https://user-images.githubusercontent.com/81224613/234163500-4befc93f-0390-4cc2-b565-f11755a12296.png">

위의 상황대로 알고리즘 처리를 마치면 list에 저장된 중점 좌표들의 평균이 리프레임 최적 중점이 됩니다. 해당 최적 중점을 통해 리프레임하면 사진에서 보이는 빨간색 박스와 같은 프레임을 얻을 수 있습니다.

**(2)-2 프레임 잘라내기**

리프레임을 할 때는 피사체를 zooming하지 않고 특정 비율에서 기존 프레임이 최대한 담길 수 있도록 설정하여 프레임이 깨지는 상황이 발생하지 않도록 하였습니다. 

예를 들어 원본 프레임의 사이즈가 1024x720이고 5:4로 잘라야 하는 경우, 원본 프레임의 비율(64/45=1.4222…)이 리프레임하려는 비율(5/4=1.25)보다 크기 때문에 원본 프레임의 넓이 부분을 잘라내어 5:4의 비율을 맞추는 방식입니다.

**(2)-3 프레임 보간**

인식된 피사체가 움직이거나, 피사체의 개수가 바뀌는 등의 경우에서는 처리하고자 하는 중점의 좌표 값이 이전 프레임에 비해 크게 달라집니다. 이때 특별한 처리 없이 그대로 잘라낸 프레임들을 송출할 경우 사용자의 입장에서 화면이 매우 흔들리는 것으로 보이게 됩니다. 따라서 프레임 보간법을 고안하였습니다.

보간이란 기존에 알던 어떠한 두 지점 사이에 위치한 새로운 데이터의 값을 추정하는 것을 의미합니다. 

<img width="302" alt="스크린샷 2022-10-04 오후 11 51 00" src="https://user-images.githubusercontent.com/81224613/234163498-3745ecd7-df4f-4ec3-887a-fc5e6b7be8ad.png">

위의 사진은 선형 보간법의 예시로, 끝점인 (x0, y0), (x1, y1)의 값이 주어졌을 때 직선 거리에 따라 선형적으로 계산하여 그 사이에 위치한 값 (x, y)를 추정하는 것입니다.

프레임 보간법은 euclidean norm으로 한 프레임의 좌표와 다른 프레임의 좌표의 직선 거리 linear하게 나타낸 후에 한 프레임의 timestamp와 다른 프레임의 timestamp 사이에 존재하는 timestamp에 해당하는 좌표 값을 추정하여 이루어집니다. 즉 한 프레임에서 다음 프레임이 송출되기 전에 보간한 좌표 값을 중심으로 crop한 프레임들을 채워 넣어서 더 자연스러운 영상이 보여지게 하는 것입니다.

<img width="252" alt="스크린샷 2022-10-05 오전 12 03 32" src="https://user-images.githubusercontent.com/81224613/234163493-8dc14c2d-e439-42e6-a759-c7defd5091db.png">

하지만 기존의 보간법을 라이브 트래킹에 사용하기에는 문제가 발생하였습니다.

euclidean norm 보간 방법을 사용할 때 Key frame들의 좌표들과 timestamp를 활용하여 보간하는데, 실시간에선 영상 데이터와 달리 다음으로 얻어오는 프레임들을 알 수 없기 때문에 해당 방법을 사용할 수 없었습니다.

따라서 이전 프레임과 현재 프레임의 중심 좌표를 비교하는 online 방식으로 프레임을 보간하였습니다. 또한 timestamp가 없는 문제는 이전 프레임과 현재 프레임 사이에 채워 넣는 보간 프레임의 개수를 임의로 정하여 해결하였습니다.



## 문제 식별

- 피사체의 위치 정보 변화에 따른 프레임 흔들림

피사체의 수가 1명에서 2명이 될 때, 혹은 처음부터 여러 명이 있을 때 프레임이 매우 흔들리는 현상이 발생하였습니다. 

인식된 피사체의 수가 2 이상일 때 임의의 범위를 두어 처리하는 알고리즘에서 설정한 범위에 어떤 피사체가 걸쳐 있을 경우 피사체를 잡았다가 안 잡는 상황이 반복되면서 프레임이 흔들리게 되는 것이었습니다. 

- object detection 모델의 정확도가 낮다.

<img width="957" alt="스크린샷 2022-10-08 오후 9 55 18" src="https://user-images.githubusercontent.com/81224613/234163488-6427bc47-ca2b-4a37-b19c-aff8fd5a820e.png">

fps를 30이상으로 구현하기 위해서 속도가 빠른 object detection 모델인 SSD를 사용하다 보니, 모델의 성능이 좋지 않았습니다. 따라서 객체를 인식하지 못하거나, 객체의 위치에 따른 bounding box를 제대로 그리지 못하는 현상이 발생하였습니다.

## 접근 방법 / 해결

이전까지는 매 프레임마다 object detection을 하여 피사체의 위치 정보를 파악하였습니다. 하지만 위의 방식이 효율적이지 않다고 생각하였습니다.

또한 fps 30이상으로 만들어야 하기 때문에 비교적 속도가 빠른 object detection 모델을 사용하였던 것인데 꼭 모델의 속도를 빨리하여 fps 30이상을 구현할 필요가 없다는 것을 알게 되었습니다.

기존 방식을 바꾸어 한 프레임에서 object detection을 하면 이후의 프레임에서는 다른 방식으로 피사체의 위치 정보를 파악하는 것을 고려하였고 이때 object tracking을 사용하기로 하였습니다.

### multiple object tracking(MOT)

multiple object tracking은 다수의 객체를 추적하는 것을 의미합니다. object tracking은 시작 프레임의 bounding box 좌표만으로 tracking하는 Detection-Free-Tracking과 object detector로 얻는 bounding box 좌표로 tracking하는 Detection-Based-Tracking으로 나뉘는데, 대부분 Detection-Based-Tracking 방식으로 사용됩니다.

또한 전체 프레임의 detection 정보를 활용하여 tracking trajactory를 만드는 batch tracking 방식과 이전 프레임과 현재 프레임의 detection 정보를 활용하여 tracking하는 online tracking 방식이 존재하는데, 실시간으로 구현해야 하는 연구 방향에 맞도록 online tracking 방식을 채택하였습니다.

object tracking에서 주로 사용되는 기술은 SORT(simple online real-time tracker)입니다.

### SORT

SORT[5]는 kalman filter와 헝가리안 알고리즘을 이용하여 피사체를 추적하는 알고리즘입니다.

- Kalman Filter

칼만 필터는 영상 내 객체의 움직임이 선형적이라고 가정하고(영상 내에서 객체가 갑자기 사라지는 경우가 매우 적기 때문), 이전 객체의 위치 및 속도를 계산하여 현재 프레임의 객체 위치를 확률적으로 추정합니다.

<img width="489" alt="스크린샷 2022-10-06 오전 11 14 42" src="https://user-images.githubusercontent.com/81224613/234163486-20f8249f-180e-45ab-af87-55658fca0e47.png">

- `u` : 이전 프레임에서의 객체 가로 중점 위치
- `v` : 이전 프레임에서의 객체 세로 중점 위치
- `s` : 객체의 bounding box의 scale
- `r` : 객체의 bounding box의 비율 (가로 / 세로)

`u`, `v`, `s`, `r` 의 값을 선형 등속 모델을 통해 현재 프레임의 객체의 상태(`u'`, `v'` , `s'`)를 추적합니다. 이때 bounding box의 비율은 일정하다고 가정합니다.

- 헝가리안 알고리즘

1) 칼만 필터를 통해 이전 프레임까지의 객체 위치를 예측합니다.

2) 현재 프레임의 객체 위치 정보를 detector로 알아냅니다.

3) 헝가리안 알고리즘을 통해 칼만 필터로 예측한 값과 detector로 인식한 값들을 거리가 가까운 것끼리 매칭합니다.

위의 과정을 통해 현재 이미지 내에서의 객체 정보를 파악하고 추적하게 됩니다.

<img width="830" alt="스크린샷 2022-10-27 오후 4 14 52" src="https://user-images.githubusercontent.com/81224613/234163485-9b01ec2b-0906-478f-a5f4-3e2f50d9608a.png">

Deep SORT[6]는 기존 SORT 방식에서 Matching cascade와 Person Re-Identification을 추가한 것입니다.

(Matching cascade와 Person Re-Identification은 부록에서 설명)

### FastMOT 사용 이유 및 원리

SORT와 DeepSORT 등의 two-stage tracker는 매 프레임마다 object detector의 detection 인식 정보를 필요로 합니다. 하지만 object detector를 더 성능이 좋은 것으로 바꾸게 된다면, 속도가 더 느린 모델을 사용해야 하므로 매 프레임마다 object detection을 할 경우에 fps가 30이상인 상태를 유지하기 어렵게 됩니다.

따라서 매 프레임마다 object detection을 하지 않아도 object tracking 하는 것이 가능한 FastMOT를 사용하였습니다.

FastMOT는 N개의 프레임마다 object detection을 하고, 그 공백을 칼만 필터를 통해 피사체의 위치를 예측하는 방식으로 작동합니다. 

칼만 필터의 예측 정확도를 더 높이기 위해 이전 프레임과 현재 프레임의 optical flow를 계산한 후에 칼만 필터를 적용하는 방식을 사용하여 object detector 없이도 tracking 프레임에서 피사체를 추적할 수 있게 됩니다.

<img width="790" alt="스크린샷 2022-08-19 오후 7 29 45" src="https://user-images.githubusercontent.com/81224613/234163483-355fb4bd-d31a-4a4e-92bb-d91e0a10d169.png">

### Sparse frame processing

첫 번째 프레임에서 object detector를 통해 피사체의 위치 정보를 인식한 후 object tracker에 해당 정보를 전달하고, 피사체의 정보를 바탕으로 잘라낸 프레임을 송출합니다. 이후 두 번째 프레임부터는 이전 detection 정보와 이전 이미지와 현재 이미지의 image feature 비교 정보를 가지고 현재 피사체의 위치 정보를 예측합니다. tracker로 추정된 위치 정보를 바탕으로 리프레임하여 프레임들을 송출하고, 해당 과정을 4번 거친 후에 다시 object detection 프레임으로 돌아오게 됩니다. 즉 detection 프레임과 tracking 프레임의 비율을 1:4로 구성하였습니다. 중간에 tracker가 인식 됐던 피사체를 놓치는 경우에는 다음 프레임에서 object detection을 하도록 하였습니다.

### 결과

위의 방식을 사용하였을 때 피사체가 여러 개인 경우에도 이전보다 프레임이 흔들리는 현상이 매우 적어졌습니다. 

매 프레임마다 detection으로 피사체의 위치 정보를 파악할 때와 달리 object tracker를 통해 인식된 피사체를 지속적으로 추적하고, 그 과정에서 각 프레임의 특징을 추출하여 사용하면서 영상 내의 흐름을 반영할 수 있게 되었기 때문입니다.

또한 object tracker로 객체를 추적할 때 처음 인식되었던 피사체를 위주로 tracking이 이루어지면서 자연스럽게 프레임이 흔들리는 현상이 줄어들었습니다.

## 결론(요약 및 향후 연구 과제)

### 요약

FastMOT object tracker를 사용하여 지속적이고 장기적으로 객체를 추적할 수 있게 하였습니다. object detection만으로 피사체를 추적하는 초기 접근 방식에서 object tracking 방식으로 피사체를 추적하는 것으로 바꾸었던 것이 연구에서 큰 성과를 보였습니다.

### 향후 연구 과제

Visual Transformer 구조에서 파생된 object detection 모델인 **[DINO](https://paperswithcode.com/paper/dino-detr-with-improved-denoising-anchor-1), [SwinV2-G](https://paperswithcode.com/paper/swin-transformer-v2-scaling-up-capacity-and)** 등 **SOTA 혹은 그에 준하는 모델**을 사용한다면 연구가 더욱 발전될 것이라고 생각됩니다. Object detection 모델의 성능이 좋아질수록 Object tracker의 성능도 좋아지기 때문에 모델의 교체가 더 큰 효과를 발휘할 것이라고 예상합니다. 다만 Object detection 모델을 교체할 경우, 프레임을 순차적으로 처리하는 기존 알고리즘대로 사용한다면 detection 프레임에서는 송출이 매우 느려지는 현상이 발생할 것입니다. 따라서 멀티프로세싱을 접목한다면 해당 문제가 해결되지 않을까 생각합니다.

## Appendix

- fps란?

fps(frame per second)는 1초당 보여지는 프레임의 개수를 의미합니다. 사람의 눈에 영상이 실시간으로 움직이는 것처럼 보이려면 fps가 30 이상이어야 합니다.

- bounding box
    
<img width="369" alt="스크린샷 2022-10-06 오전 11 35 35" src="https://user-images.githubusercontent.com/81224613/234163482-a1136541-cd07-4250-8df8-282ee5d91f27.png">
    

Object detection에서 객체의 위치를 표현할 때 사용하는 직사각형 모양의 박스입니다.

x축과 y축을 통해 표현하며, (x1, y1)은 x값과 y값의 최솟값을 나타내고 (x2, y2)은 x,y 값의 최댓값을 나타냅니다.

- mAP score

mAP(mean Average Precision)는 Object detector의 정확도를 평가할 때 주로 사용되는 평가지표입니다. mAP에 대해 알기 전에 IOU, precision, recall개념에 대해 알아야 합니다.

Object detection에서 객체의 위치를 맞게 추정했는지 판단할 때 IOU라는 개념을 사용합니다. 

<img width="583" alt="스크린샷 2022-10-08 오후 10 30 00" src="https://user-images.githubusercontent.com/81224613/234163480-8b9352ee-06f6-4f38-8e62-65f61c822e34.png">

IOU(Intersection over Union)는 교집합의 영역 / 합집합의 영역입니다.

Object detector가 예측한 bounding box와 실제 객체의 bounding box(ground-truth) 영역 간의 IOU 값이 0.5 이상일 때 Object detector가 옳게 예측했다고 판단하게 됩니다.

<img width="308" alt="스크린샷 2022-10-08 오후 10 16 47" src="https://user-images.githubusercontent.com/81224613/234163478-621af6fe-ffb7-462b-b7ee-e3e2384f7a0d.png">

- $precision = { TP \over TP + FP }$

precision(정밀도)은 모델이 True라고 예측한 것들 중에서 실제로 True인 것의 비율입니다.

- $recall = { TP \over TP + FN }$

recall(재현률)은 실제로 True인 것들 중에서 모델이 True라고 한 것의 비율입니다.

둘 중 한 값만으로는 좋은 모델을 판단하기 어렵기 때문에 precision과 recall을 동시에 사용하여 모델을 평가하게 됩니다. precision과 recall 값으로 그린 그래프의 아래 면적을 AP(Average Precision)로 정의하고 각 클래스에서 구한 AP의 평균인 mAP를 통해 Object detector의 성능을 평가합니다.

- DeepSORT
    - Matching cascade
    
    최근에 생성된 track에 우선 순위를 부여하는 알고리즘입니다. 가장 최근에 생성된 track일수록 더 정확한 추적이고, 나중에 생성된 track일수록 불확실한 추적이라고 판단하기 때문입니다. 
    

<img width="450" alt="스크린샷 2022-10-09 오후 3 18 12" src="https://user-images.githubusercontent.com/81224613/234163471-22cb8cdb-999a-4e82-bf85-3f2468ca9e15.png">
    
* Matching Cascade가 일어나는 과정은 다음과 같습니다.
    
    - Track과 Detection 간의 cost matrix를 구합니다. cost matrix는 Mahalanobis distance와 cosine distance를 통해 구하는데, 논문에서는 cosine distance만을 사용했을 때 오히려 성능이 좋다고 합니다.
    - track에 대한 추정 타당성을 검증하는 gate matrix를 구합니다. Track과 Detection 사이의 cost에 임계값을 설정하는 방법을 통해 있을 것 같지 않다고 판단되는 track을 제거합니다.
    - matched detections를 공집합으로, unmatched detections를 Detections로 초기화합니다.
    - 각 age를 가지는 track들을 순차적으로 탐색하면서, linear assignment를 진행합니다. 이때 matching되지 않은 track들은 제거합니다.
    
    - Re-identification
    
<img width="288" alt="스크린샷 2022-10-06 오후 3 31 27" src="https://user-images.githubusercontent.com/81224613/234163515-e2d691e3-a2dc-4b8b-b866-facdf44a788d.png">

<img width="887" alt="스크린샷 2022-10-06 오후 3 31 52" src="https://user-images.githubusercontent.com/81224613/234163513-0c59acdb-46f3-4965-90f4-4bf6beeef08a.png">

    SORT는 ID switching의 문제를 갖고 있습니다. ID switching은 다양한 객체를 추적할 때, 각 개체의 track ID가 바뀌는 현상입니다. ID switching은 tracking의 성능을 저하시키는 요인이기 때문에 이를 해결하고자 Re-identification 모델을 적용하였습니다. 
    
    Person Re-identification은 특정 사람을 다양한 각도나 위치에 있는 다른 이미지들에서 찾는 task입니다. CNN의 feature space 상에서 동일한 사람에 대한 feature는 feature 사이의 거리가 가깝게 mapping하고, 다른 사람에 대한 feature는 feature 사리의 거리가 멀게 mapping하는 방식을 통해 사람의 특징을 잘 파악하는 모델을 얻을 수 있고, 이 모델이 OSNet입니다. 
    
    DeepSORT에서는 OSNet을 활용하여 detection된 피사체를 잘라낸 프레임에서 image feature를 추출합니다. 해당 정보를 tracking에 사용하여 ID switching 문제를 45% 감소시켰습니다.
    

## Reference

[1] Google Research, "AutoFlip: An Open Source Framework for Intelligent Video Reframing", [https://ai.googleblog.com/2020/02/autoflip-open-source-framework-for.html](https://ai.googleblog.com/2020/02/autoflip-open-source-framework-for.html) (accessed Oct. 06, 2022)

[2] Apple (2021), "Capture high-quality photos using video formats", [https://developer.apple.com/videos/play/wwdc2021/10047/?time=808](https://developer.apple.com/videos/play/wwdc2021/10047/?time=808) 

[3] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg (2016), "SSD: Single Shot MultiBox Detector", [https://arxiv.org/pdf/1512.02325.pdf](https://arxiv.org/pdf/1512.02325.pdf)

[4] Google Research (2019), "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs", [https://arxiv.org/pdf/1907.05047.pdf](https://arxiv.org/pdf/1907.05047.pdf)

[5] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft (2016), "Simple Online and Realtime Tracking", [https://arxiv.org/pdf/1602.00763.pdf](https://arxiv.org/pdf/1602.00763.pdf)

[6] Nicolai Wojke, Alex Bewley, Dietrich Paulus (2017), "Simple Online and Realtime Tracking with a Deep Association Metric", [https://arxiv.org/pdf/1703.07402.pdf](https://arxiv.org/pdf/1703.07402.pdf)
