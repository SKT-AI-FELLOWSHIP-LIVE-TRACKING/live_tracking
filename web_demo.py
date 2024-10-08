from logging import raiseExceptions
import cv2
import argparse
import timeit
import asyncio
import numpy as np
from dtos import DetectionRegions, FaceRegions, FMOT_TrackingRegions
from face_detection import face_detection
from object_detection import object_detection
from detection_processing import detection_processing
from utils import *
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


# [t, l, b, r], class_id, score
# [t, l, b, r] == [x1,y1,x2,y2]

def regions_to_detections_BYTE(all_regions):
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
        boxes.append((x1, y1, x2, y2, score))

    return np.array(boxes)


def detect_objects(image):
  # face detection
  fd = face_detection(image)
  fd.detect_faces()
  regions = fd.localization_to_region()
  # visualize_faces(image, regions, image_width, image_height)

  # object detection
  od = object_detection(image)
  output_dict, category_index = od.detect_objects()
  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']
  # visualize_objects(image, boxes, classes, scores, category_index, image_width, image_height)     


  # detection processing
  dp = detection_processing(boxes, classes, scores, regions[0])
  dp.tensors_to_regions()
  all_regions = dp.sort_detection()

  return all_regions


def x_processing(all_regions, scaled_target_width, pre_x_center):
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
    if (i == 0): 
      x_center_list.append(x_center)
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
    optimal_x_center = pre_x_center
  
  return optimal_x_center


def y_processing(all_regions, scaled_target_height, pre_y_center):
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
    if (i == 0): 
      y_center_list.append(y_center)
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
    optimal_y_center = pre_y_center
  
  return optimal_y_center

# Interpolation function
def interpolate_coords(old_center, new_center, alpha):
    return (1 - alpha) * np.array(old_center) + alpha * np.array(new_center)

# Calculate alpha based on FPS
def calculate_alpha(fps):
    return min(1.0, max(0.01, 1.0 / fps))


async def main(config):
  # For webcam input:
  cap = cv2.VideoCapture(0) # webcam -> 0
  global image_width, image_height 
  image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # 비율 정하기
  original_ratio = get_ratio(image_width, image_height)
  requested_ratio = config.w / config.h
  target_width, target_height, scaled_target, wh_flag = decide_target_size(original_ratio, requested_ratio, image_width, image_height)

  pre_x_center = 0.5
  pre_y_center = 0.5
  fps = 30
  frame_id = 0
  all_regions = []

  tracker = BYTETracker(config)


  while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      start_t = timeit.default_timer()

      outputs = None

      if (frame_id  >= 1): # detection per 5 frames # % 5 == 0
        # detect objects
        all_regions = detect_objects(image)
        dets = regions_to_detections_BYTE(all_regions)
        if (len(dets)):
          outputs = tracker.update(dets, [image_height, image_width], (image_height, image_width))
  


      if outputs is not None:
        results = []
        for t in outputs:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 0.2 and not vertical:
              xmin = tlwh[0] / image_width
              ymin = tlwh[1] / image_height
              w = tlwh[2] / image_width
              h = tlwh[3] / image_height

              # img = image[int(tlwh[1]):int(tlwh[1]+tlwh[3]),int(tlwh[0]):int(tlwh[0]+tlwh[2])]
              try:
                  results.append(FMOT_TrackingRegions(xmin, ymin, w, h, tid))
              except:
                  print("Failed to update TrackingRegions")
        
        # save results
        all_regions = results

      # print(all_regions)

      # Reframe
      if (wh_flag == 0):
        optimal_x_center = x_processing(all_regions, scaled_target, pre_x_center)

        terminate_t = timeit.default_timer()
        if (abs(pre_x_center - optimal_x_center) * image_width < 10): # 가만히 있을 때 흔들리는 이슈 해결하기 위해 사용 / 현재 10으로 흔들리는 것을 방지해놨는데 이보다 크게 파라미터를 설정하면 interpolation 과정에서 프레임이 불안정하게 보간됨.
          optimal_x_center = pre_x_center
        # await real_time_interpolate_x(pre_x_center, optimal_x_center, image_width, target_width, image)
        
        alpha = calculate_alpha(fps) # alpha 계산 based on current FPS
        optimal_x_center = interpolate_coords(pre_x_center, optimal_x_center, alpha) # linear interpolation -> optimal coords update
        
        left = int(optimal_x_center * image_width - target_width / 2)
        if (left < 0):
          left = 0
        elif (left > image_width - target_width):
          left = image_width - target_width

        pre_x_center = optimal_x_center
        img = image[:, left:left+target_width]

      else: # 세로가 crop될 때
        optimal_y_center = y_processing(all_regions, scaled_target, pre_y_center)

        terminate_t = timeit.default_timer()
        if (abs(pre_y_center - optimal_y_center) * image_height < 10): 
          optimal_y_center = pre_y_center

        alpha = calculate_alpha(fps) # alpha 계산 based on current FPS
        optimal_y_center = interpolate_coords(pre_y_center, optimal_y_center, alpha) # linear interpolation -> optimal coords update
        
        top = int(optimal_y_center * image_height - target_height / 2)
        if (top < 0):
          top = 0
        elif (top > image_height - target_height):
          top = image_height - target_height

        pre_y_center = optimal_y_center
        img = image[top:top+target_height, :]

      frame_id += 1

      img = img.copy()
      # fps 계산
      terminate_t = timeit.default_timer()
      fps = int(1.0 / (terminate_t - start_t))
      cv2.putText(img,
                  "FPS:" + str(int(fps)),
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
  parser.add_argument('--w', type=float, default=1, help='Ratio of Frame Width')
  parser.add_argument('--h', type=float, default=1, help='Ratio of Frame Height')
  parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
  parser.add_argument(
    "-f",
    "--exp_file",
    default=None,
    type=str,
    help="pls input your expriment description file",
  )
  parser.add_argument(
    "--fp16",
    dest="fp16",
    default=False,
    action="store_true",
    help="Adopting mix precision evaluating.",
  )
  parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
  parser.add_argument("--conf", default=0.01, type=float, help="test conf")
  parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
  parser.add_argument("--tsize", default=None, type=int, help="test img size")
  parser.add_argument("--seed", default=None, type=int, help="eval seed")
  parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
  parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
  parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
  parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
  parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
  config = parser.parse_args()
  print(config)
  asyncio.run(main(config))