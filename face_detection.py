import cv2
import math
import mediapipe as mp
from mediapipe.framework.formats.detection_pb2 import Detection
from dtos import FaceRegions
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils



class face_detection:
    def __init__(self, image):
        self.image = image
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.RELATIVE_BOUNDING_BOX = 2
        self.mp_face_detection = mp.solutions.face_detection
        self.regions = []

    #detect faces
    def detect_faces(self):
        """
        Function to detect faces
        -------
        Returns:
            NONE
        """
        # model_selction = 0 -> 속도 빠른 모델, 1 -> 정확도 높은 속도 느린 모델
        # 모델 불러와서 face detect
        with self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6) as face_detection:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.results = face_detection.process(self.image)

            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            self.detections = self.results.detections
    
    # 다른 노드에 디텍션 정보 종합하기 위함
    # localization_to_region
    # dtos.FaceRegions
    def localization_to_region(self):
        """
        Function to detect objects
        -------
        Returns
            face_full : list
                face regions detected 
            core_landmark : list, optional
                key landmarks of face (eyes, nose, mouth)
            all_landmark : list, optional
                all landmarks of face (including ears)
        """
        # 얼굴이 없는 경우도 고려해야 함.
        self.face_full = []
        self.core_landmark = []
        self.all_landmark = []

        if self.detections:
            for detection in self.detections:

                location = detection.location_data
                box = detection.location_data.relative_bounding_box

                # error 발생시켜야 함
                if(location.format != self.RELATIVE_BOUNDING_BOX):
                    print(location.format)
                    print("Face detection input is lacking required relative_bounding_box()")

                # error 발생시켜야 함
                # if(detection.location_data.relative_keypoints.size != 6):
                #     print("Face detection input expected 6 keypoints, has", location.relative_keypoints.size)
                

                # detection된 얼굴이 여러개인 경우 고려하여
                # 한 클래스를 한 명의 디텍션 정보로 구성하고 이를 리스트로 묶음.

                # face_full
                #region = FaceRegions()

                x = max(float(0), box.xmin)
                y = max(float(0), box.ymin)
                width = min(box.width - abs(x - box.xmin), 1 - x)
                height = min(box.height - abs(y - box.ymin), 1 - y)
                SignalType = "FACE_FULL"
                visual_score = detection.score[0]

                region = FaceRegions(x, y, width, height, SignalType, visual_score)
                # region.update_region(x, y, width, height, SignalType, visual_score)
                self.face_full.append(region)


                # landmark 총 6개 -> 각 랜드마크 당 클래스 하나
                # core는 앞에 4개만
                core_landmark_region = []
                all_landmark_region = []
                
                for i in range(4):
                    keypoint = location.relative_keypoints[i]
                    x = keypoint.x
                    y = keypoint.y
                    w = self.normalize_x(1)
                    h = self.normalize_y(1)
                    SignalType = "FACE_LANDMARK"
                    
                    region = FaceRegions(x, y, w, h, SignalType, visual_score)
                    region = self.extend_salient_region_with_point(keypoint.x, keypoint.y, region)

                    core_landmark_region.append(region)
                    all_landmark_region.append(region)

                
                for i in range(4, 6):
                    keypoint = location.relative_keypoints[i]
                    x = keypoint.x
                    y = keypoint.y
                    w = self.normalize_x(1)
                    h = self.normalize_y(1)
                    SignalType = "FACE_LANDMARK"
                   
                    region = FaceRegions(x, y, w, h, SignalType, visual_score)

                    region = self.extend_salient_region_with_point(keypoint.x, keypoint.y, region)

                    all_landmark_region.append(region)


                self.core_landmark.append(core_landmark_region)
                self.all_landmark.append(all_landmark_region)
                
        
        return self.face_full, self.core_landmark, self.all_landmark

    def extend_salient_region_with_point(self, x, y, region):
        """
        Function to extend regions in consideration of error
        -------
        Parameters
            x : float
                x of face keypoint
            y : float
                y of face keypoint
            region : dataclass
                class containing face regions
        -------
        Returns
            region : dataclass
                updated regions
            
        """
        if(region.w is None):
            region.set_w(self.normalize_x(1))
        elif(x < region.x):
            region.set_w(region.w + region.x - x)
        elif(x > region.x + region.w):
            region.set_w(x - region.x)
        
        if(region.h is None):
            region.set_h(self.normalize_y(1))
        elif(y < region.y):
            region.set_h(region.h + region.y - y)
        elif(y > region.y + region.h):
            region.set_h(y - region.y)

        if(region.x is None):
            region.set_x(x)
        else:
            region.set_x(min(x, region.x))
        
        if(region.y is None):
            region.set_y(y)
        else:
            region.set_y(min(y, region.y))

        return region

    def normalize_x(self, pixel):
        return pixel / float(self.image_width)

    def normalize_y(self, pixel):
        return pixel / float(self.image_height)

        
