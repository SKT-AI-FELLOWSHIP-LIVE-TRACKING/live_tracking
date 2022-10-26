from logging import raiseExceptions
import re
from urllib import request
import cv2
import math
import argparse
import timeit
import asyncio
import numpy as np
import mediapipe as mp
from reid.torchreid.utils import FeatureExtractor
from dtos import DetectionRegions
from dtos import FaceRegions
from dtos import FMOT_TrackingRegions
from face_detection import face_detection
from object_det_v2 import object_detection
from detection_processing import detection_processing
from FastMOT.fastmot.tracker import MultiTracker
from FastMOT.fastmot.utils import ConfigDecoder
import json
from types import SimpleNamespace
from utils import *

# FastMOT는 현재 사람만 트래킹! -> 각 label 추가할 것.

# 세로로 자르려면?
# 변수 수정 -> pre_x_center 등
# 함수 수정 -> list: y_center 등

# detections type
DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


# [t, l, b, r], class_id, score
### FastMOT는 PIL 사용 -> [t, l, b, r] == [x1,y1,x2,y2]

def regions_to_detections(all_regions):
    boxes = []
    d = DetectionRegions(0,0,0,0,0,-1)
    f = FaceRegions(0,0,0,0,'tmp',0)
    t = FMOT_TrackingRegions(0,0,0,0,-1)
    for i, region in enumerate(all_regions):
        y1 = int(region.y * image_height)
        x1 = int(region.x * image_width)
        y2 = int((region.y + region.h) * image_height)
        x2 = int((region.x + region.w) * image_width)
        
        if (type(region) == type(f)):
            class_id = 1
        elif (type(region)== type(d)):
            class_id = int(region.class_id)
        else:
            raiseExceptions("data type을 확인할 수 없습니다.")
        score = region.score
        # boxes.append(([top, left, bottom, right], class_id, score))
        boxes.append(([x1, y1, x2, y2], class_id, score))

    return np.array(boxes, DET_DTYPE).view(np.recarray)

def detect_objects(image):
  # face detection
  fd = face_detection(image)
  fd.detect_faces()
  regions = fd.localization_to_region()
  #visualize_faces(image, regions, image_width, image_height)

  # object detection
  od = object_detection(image)
  output_dict, category_index = od.detect_objects()
  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']
  #visualize_objects(image, boxes, classes, scores, category_index, image_width, image_height)     


  # detection processing
  dp = detection_processing(boxes, classes, scores, regions[0])
  dp.tensors_to_regions()
  all_regions = dp.sort_detection()

  return all_regions


def x_processing(all_regions, scaled_target_width):
  # detection 없을 때 고려해야 함
  # detection 여러 개일 때 프레임 흔들림 (두 개 model -> 잡을 때와 안 잡힐 때)
  # 얼굴을 detection 했을 때는 전적으로 신뢰하기로 ㄱㄱ
  x_center_list = []
  optimal_x_center = 0
  scaled_target = scaled_target_width
  for i, region in enumerate(all_regions):
    x = region.x
    y = region.y
    w = region.w
    h = region.h
    x_center = x + w / 2
    y_center = y + h / 2
    if (i == 0): # 가로 기준(세로 고정) 나중에 수정해야 함 + 기준은 float로
      x_center_list.append(x_center)
      # score_list.append(x_center)
      min_ = max_ = x_center_list[0]
    elif (scaled_target <= 0):
      break
    elif ((min_ - x_center > 0) and (min_ - x_center < scaled_target)):
      x_center_list.append(x_center)
      scaled_target -= (min_ - x_center)
      min_ = x_center
    elif ((x_center - max_ < scaled_target) and (x_center - max_ > 0)):
      x_center_list.append(x_center)
      scaled_target -= (x_center - max_)
      max_ = x_center

  if (len(x_center_list)):
    optimal_x_center = np.average(x_center_list)
  else:
    optimal_x_center = 0.5
  
  return optimal_x_center


def y_processing(all_regions, scaled_target_height):
  y_center_list = []
  optimal_y_center = 0
  scaled_target = scaled_target_height
  for i, region in enumerate(all_regions):
    x = region.x
    y = region.y
    w = region.w
    h = region.h
    x_center = x + w / 2
    y_center = y + h / 2
    if (i == 0): # 가로 기준(세로 고정) 나중에 수정해야 함 + 기준은 float로
      y_center_list.append(y_center)
      # score_list.append(x_center)
      min_ = max_ = y_center_list[0]
    elif (scaled_target <= 0):
      break
    elif ((min_ - y_center > 0) and (min_ - y_center < scaled_target)):
      y_center_list.append(y_center)
      scaled_target -= (min_ - y_center)
      min_ = y_center
    elif ((y_center - max_ < scaled_target) and (y_center - max_ > 0)):
      y_center_list.append(y_center)
      scaled_target -= (y_center - max_)
      max_ = y_center

  if (len(y_center_list)):
    optimal_y_center = np.average(y_center_list)
  else:
    optimal_y_center = 0.5
  
  return optimal_y_center

      

async def main(config):
  # For webcam input:
  cap = cv2.VideoCapture(0) # webcam -> 0
  global image_width, image_height 
  image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # 비율 정하기
  original_ratio = get_ratio(image_width, image_height)
  # requested_ratio = 9 / 16 # arg 인자로 받아오기 !
  requested_ratio = config.w / config.h
  target_width, target_height, scaled_target, wh_flag = decide_target_size(original_ratio, requested_ratio, image_width, image_height)

  pre_x_center = 0.5
  pre_y_center = 0.5
  fps = 0
  frame_id = 0

  mot_json = "FastMOT/cfg/mot.json"
  with open(mot_json) as cfg_file:
    mot_json = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

  tracker = MultiTracker((image_width, image_height), 'cosine', **vars(mot_json.mot_cfg.tracker_cfg))
  frame_rate = 30
  cap_dt = 1. / frame_rate
  tracker.reset(cap_dt)

  # initialize deep sort
  model_name = "osnet_x0_25"
  model_weights = "osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth"
  feature_extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_weights,
            device='cpu'
  )

  while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      start_t = timeit.default_timer()

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


      if (frame_id == 0):
        # detect objects
        all_regions = detect_objects(image)

        ### image color transition
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #### detections 처리
        detections = regions_to_detections(all_regions)

        # tracker initiation
        tracker.init(image, detections)

        ### image color transition
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


      elif (frame_id % 5 == 0): # detection per 5 frames # % 5 == 0
        # detect objects
        all_regions = detect_objects(image)

        ### image color transition
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tracker.compute_flow(image)

        # track
        tracker.apply_kalman()

        ############################
        detections = regions_to_detections(all_regions)

        features = get_features(detections.tlbr, image, feature_extractor)

        if (len(features)):
            embeddings = features.numpy()
            ### 디텍션 처리
            tracker.update(frame_id, detections, embeddings)

            results = []
            track_lists = list(track for track in tracker.tracks.values()
                    if track.confirmed and track.active)
            for track in track_lists:
                bbox = track.tlbr
                xmin = bbox[0] / image_width
                ymin = bbox[1] / image_height
                w = (bbox[2] - bbox[0]) / image_width
                h = (bbox[3] - bbox[1]) / image_height

                try:
                    results.append(FMOT_TrackingRegions(xmin, ymin, w, h, track.trk_id))
                except:
                    print("Failed to append Tracking Regions")
            
            if (len(results) == 0):
                frame_id = 4
            else:
                all_regions = results
        
        # no detection
        else:
            all_regions = []

        ### image color transition
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        ############################

      else:
        ### image color transition
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # tracking
        tracker.track(image)

        track_lists = list(track for track in tracker.tracks.values()
                if track.confirmed and track.active)
        results = []
        for track in track_lists:
            if not (track.confirmed and track.active):
                continue
            bbox = track.tlbr

            xmin = bbox[0] / image_width
            ymin = bbox[1] / image_height
            w = (bbox[2] - bbox[0]) / image_width
            h = (bbox[3] - bbox[1]) / image_height

            try:
                results.append(FMOT_TrackingRegions(xmin, ymin, w, h, track.trk_id))
            except:
                print("Failed to update TrackingRegions")
        if (len(results) == 0):
            frame_id = 4
        else:
            all_regions = results

        ### image color transition
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      frame_id += 1

      print(all_regions)

      ############################
      # # detection 없을 때 고려해야 함
      # # detection 여러 개일 때 프레임 흔들림 (두 개 model -> 잡을 때와 안 잡힐 때)
      # # 얼굴을 detection 했을 때는 전적으로 신뢰하기로 ㄱㄱ
      # x_center_list = []
      # # score_list = []
      # optimal_x_center = 0
      # scaled_target = scaled_target_width
      # for i, region in enumerate(all_regions):
      #   x = region.x
      #   y = region.y
      #   w = region.w
      #   h = region.h
      #   x_center = x + w / 2
      #   y_center = y + h / 2
      #   if (i == 0): # 가로 기준(세로 고정) 나중에 수정해야 함 + 기준은 float로
      #     x_center_list.append(x_center)
      #     # score_list.append(x_center)
      #     min_ = max_ = x_center_list[0]
      #   elif (scaled_target <= 0):
      #     break
      #   elif ((min_ - x_center > 0) and (min_ - x_center < scaled_target)):
      #     x_center_list.append(x_center)
      #     scaled_target -= (min_ - x_center)
      #     min_ = x_center
      #   elif ((x_center - max_ < scaled_target) and (x_center - max_ > 0)):
      #     x_center_list.append(x_center)
      #     scaled_target -= (x_center - max_)
      #     max_ = x_center

      
      # if (len(x_center_list)):
      #   optimal_x_center = np.average(x_center_list)
      #   # print(x_center_list)
      # else:
      #   optimal_x_center = 0.5

      if (wh_flag == 0):
        optimal_x_center = x_processing(all_regions, scaled_target)

        terminate_t = timeit.default_timer()
        if (abs(pre_x_center - optimal_x_center) * image_width < 10): # 가만히 있을 때 흔들리는 이슈 해결하기 위해 사용.... 일시적인 방법이므로 이보다 더 좋은 방법이 있을 시 대체하는 것이 좋을듯. 현재 10으로 흔들리는 것을 방지해놨는데 이보다 크게 파라미터를 설정하면 interpolation 과정에서 프레임이 불안정하게 보간됨.
          optimal_x_center = pre_x_center
        await real_time_interpolate_x(pre_x_center, optimal_x_center, image_width, target_width, image)
        # terminate_t = timeit.default_timer()
        
        left = int(optimal_x_center * image_width - target_width / 2)
        if (left < 0):
          left = 0
        elif (left > image_width - target_width):
          left = image_width - target_width

        pre_x_center = optimal_x_center
        img = image[:, left:left+target_width]

      else: # 세로가 crop될 때
        optimal_y_center = y_processing(all_regions, scaled_target)

        terminate_t = timeit.default_timer()
        if (abs(pre_y_center - optimal_y_center) * image_height < 10): # 가만히 있을 때 흔들리는 이슈 해결하기 위해 사용.... 일시적인 방법이므로 이보다 더 좋은 방법이 있을 시 대체하는 것이 좋을듯. 현재 10으로 흔들리는 것을 방지해놨는데 이보다 크게 파라미터를 설정하면 interpolation 과정에서 프레임이 불안정하게 보간됨.
          optimal_y_center = pre_y_center
        await real_time_interpolate_y(pre_y_center, optimal_y_center, image_height, target_height, image)
        # terminate_t = timeit.default_timer()
        
        top = int(optimal_y_center * image_height - target_height / 2)
        if (top < 0):
          top = 0
        elif (top > image_height - target_height):
          top = image_height - target_height

        pre_y_center = optimal_y_center
        img = image[top:top+target_height, :]

      # fps 계산
      # terminate_t = timeit.default_timer()
      fps += int(1.0 / (terminate_t - start_t))
      cv2.putText(img,
                  "FPS:" + str(int(fps / (frame_id+1))),
                  (20, 60),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  2,
                  (0, 0, 255),
                  2)

      cv2.imshow('cropped', img)

      if cv2.waitKey(10) & 0xFF == 27:
        break
  cap.release()
  cv2.destroyAllWindows()
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default="default", help='If u want sport tracking, set "--mode sport".')
  parser.add_argument('--w', type=int, default=1, help='Ratio of Frame Width')
  parser.add_argument('--h', type=int, default=1, help='Ratio of Frame Height')
  config = parser.parse_args()
  print(config)
  asyncio.run(main(config))