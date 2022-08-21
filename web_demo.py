import re
import cv2
import math
# import argparse
import timeit
import asyncio
import numpy as np
import mediapipe as mp
from face_detection import face_detection
from object_det_v2 import object_detection
from detection_processing import detection_processing

def get_ratio(image_width, image_height):
  return image_width / image_height

def round_to_even(value):
  rounded_value = round(value)

  if (rounded_value % 2 == 1):
    rounded_value = max(2, rounded_value - 1)
  
  return rounded_value


def visualize_faces(image, regions, image_width, image_height):
  """
  face detection visualization
  ----------
  Parameters
    image : Array of uint8
    regions : list
      0 : face full region. Shape (N)
      1 : face core landmarks. Shape (N, 4)
      2 : face all landmarks. Shape (N, 6)
    image_width : int
    image_height : int
  -------
  Returns
    NONE
  """
  if regions[0] != []:
      for i in range(len(regions[0])):
        face_full_region = [regions[0][i].x, regions[0][i].y, regions[0][i].w, regions[0][i].h]

        x_px = min(math.floor(face_full_region[0] * image_width), image_width - 1)
        y_px = min(math.floor(face_full_region[1] * image_height), image_height - 1)
        w_px = min(math.floor(face_full_region[2] * image_width), image_width - 1)
        h_px = min(math.floor(face_full_region[3] * image_height), image_height - 1)
        cv2.rectangle(image, (x_px, y_px), (x_px+w_px, y_px+h_px), (0,0,255), 3)

        score = regions[0][i].score
        score = round(score, 5)
        cv2.putText(image,
            "FACE: " + str(score),
            (x_px, y_px + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2)

      face_all_landmark = regions[2]
      for i in range(len(face_all_landmark)):
        all_landmark = face_all_landmark[i]
        for j in range(6):

          x_px = min(math.floor(all_landmark[j].x * image.shape[1]), image.shape[1] - 1)
          y_px = min(math.floor(all_landmark[j].y * image.shape[0]), image.shape[0] - 1)
          w_px = int(all_landmark[j].w * image.shape[1])
          h_px = int(all_landmark[j].h * image.shape[0])

          cv2.rectangle(image, (x_px, y_px), (x_px+w_px, y_px+h_px), (255,255,255), 3)

def visualize_objects(image, boxes, classes, scores, category_index, image_width, image_height):
  """
  object detection visualization
  ----------
  Parameters
    image : Array of uint8
    regions : list
    boxes : list
      local information of detected objects. Shape (N, 4) 
    classes : list
      classes of detected objects. Shape (N)
    scores : list
      scores of detected objects. Shape (N)
    category_index : dict
    image_width : int
    image_height : int
  -------
  Returns
    NONE
  """
  required_categories = [0, # person
                         15,16,17,18,19,20,21,22,23,24, # 동물
                         36] # sports ball
  
  for i in range(len(classes)):
    if (binary_search(required_categories, classes[i]) == False):
      continue
    ymin, xmin, ymax, xmax = boxes[i]
    (left, right, top, bottom) = (xmin * image_width, xmax * image_width, ymin * image_height, ymax * image_height)
    left = int(left)
    right = int(right)
    top = int(top)
    bottom = int(bottom)

    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    cv2.putText(image,
            str(category_index[classes[i]]['name']),
            (left, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            2)
    cv2.putText(image,
            str(scores[i]),
            (left, top + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            2)

def binary_search(array, search):
  if (len(array) == 1):
    if (array[0] == search):
        return True
    else:
        return False
  if (len(array) == 0):
      return False
  
  median = len(array) // 2
  if (search == array[median]):
    return True
  if (search > array[median]):
    return binary_search(array[median:], search)
  else:
    return binary_search(array[:median], search)

def decide_target_size(original_ratio, requested_ratio, image_width, image_height):
  if (original_ratio > requested_ratio):
    target_height = round_to_even(image_height)
    target_width = round_to_even(image_height * requested_ratio)
    scaled_target_width = target_width / image_width / 2

    return target_width, target_height, scaled_target_width
  
  else:
    target_width = round_to_even(image_width)
    target_height = round_to_even(image_width / requested_ratio)
    scaled_target_height = target_height / image_height

    return target_width, target_height, scaled_target_height / 2

class piecewise_func():
  def __init__(self, start, end):
    self.start_x = 0
    self.start_y = start
    self.end_x = 30
    self.end_y = end
    self.time_ = 30 # fps 30 -> 1 sec
  
  def evaluate(self, input):
    return self.end_x - (self.end_x - input) / (self.end_x - self.start_x) * (self.end_y - self.start_y)

def show_fps(img, start_t):
  terminate_t = timeit.default_timer()
  fps = int(1.0 / (terminate_t - start_t))
  cv2.putText(img,
              "FPS:" + str(fps),
              (20, 60),
              cv2.FONT_HERSHEY_SIMPLEX,
              2,
              (0, 0, 255),
              2)
  return img

def float_frame_imshow(interpolated, image_width, target_width, image, start_t):
  # x_center = min(math.floor(interpolated * image_width), image_width - 1)
  x_center = int(interpolated * image_width)
  left = int(x_center - target_width / 2)
  if (left < 0):
    left = 0
  elif (left > image_width - target_width):
    left = image_width - target_width
  img = image[:, left:left + target_width]
  # img = show_fps(img, start_t)
  cv2.imshow('cropped', img)


# test
# 카메라 기준 1초
async def real_time_interpolate(pre_x_center, optimal_x_center, image_width, target_width, image, start_t):
  time_ = 30 # fps 30
  start = pre_x_center
  end = optimal_x_center
  func = piecewise_func(start, end)
  for i in range(1, time_):
    interpolated = func.evaluate(i)
    if (int(interpolated * image_width) == optimal_x_center):
      break
    float_frame_imshow(interpolated, image_width, target_width, image, start_t)
    # print("INTERPOLATING")
    







async def camera_sweeping(left, pre_x_center, optimal_x_center, image_width, target_width, image):
  if (pre_x_center > optimal_x_center):
    idx = -1
  else:
    idx = 1
  sweep = left + pre_x_center - optimal_x_center
  print("SWEEP", sweep, left, pre_x_center, optimal_x_center)
  while (sweep != left):
    sweep += idx
    print("SWEPPING", sweep, pre_x_center, optimal_x_center)
    if (sweep < 0 or sweep > image_width - target_width):
      break
    img = image[:, sweep:sweep+target_width]
    cv2.imshow('cropped', img)
    
# async def show_cropped_frame()

# async def show_raw_frame()

async def main():
  # For webcam input:
  cap = cv2.VideoCapture(0)
  pre_x_center = 0.5
  last_detection = 0

  while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
      
      image_width = image.shape[1]
      image_height = image.shape[0]

      start_t = timeit.default_timer()

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # face detection
      fd = face_detection(image)
      fd.detect_faces()
      regions = fd.localization_to_region()
      visualize_faces(image, regions, image_width, image_height)

      # object detection
      od = object_detection(image)
      output_dict, category_index = od.detect_objects()
      boxes = output_dict['detection_boxes']
      classes = output_dict['detection_classes']
      scores = output_dict['detection_scores']
      visualize_objects(image, boxes, classes, scores, category_index, image_width, image_height)     
      
      # detection processing
      dp = detection_processing(boxes, classes, scores, regions[0])
      dp.tensors_to_regions()
      all_regions = dp.sort_detection()
      print(all_regions)


      ### 예비 구현

      original_ratio = get_ratio(image_width, image_height)
      requested_ratio = 5 / 4
      target_width, target_height, scaled_target = decide_target_size(original_ratio, requested_ratio, image_width, image_height)


      # detection 없을 때 고려해야 함
      # detection 여러 개일 때 프레임 흔들림 (두 개 model -> 잡을 때와 안 잡힐 때)
      # 얼굴을 detection 했을 때는 전적으로 신뢰하기로 ㄱㄱ
      x_center_list = []
      score_list = []
      optimal_x_center = 0
      for i, region in enumerate(all_regions):
        x = region.x
        y = region.y
        w = region.w
        h = region.h
        x_center = x + w / 2
        y_center = y + h / 2
        if (i == 0): # 가로 기준(세로 고정) 나중에 수정해야 함 + 기준은 float로
          x_center_list.append(x_center)
          score_list.append(x_center)
          min_ = max_ = x_center_list[0]
        elif (scaled_target <= 0):
          break
        elif ((min_ - x_center > 0) and (min_ - x_center < scaled_target)):
          scaled_target -= min_ - x_center
          min_ = x_center
        elif ((x_center - max_ < scaled_target) and (x_center - max_ > 0)):
          scaled_target -= x_center - max_
          max_ = x_center

        # elif (x_center_list[0] - x_center < scaled_target): # 바운더리 수정
        #   x_center_list.append(x_center)
        #   score_list.append(x_center)
        #   scaled_target -= x_center_list[0] - x_center
      
      if (len(x_center_list)):
        optimal_x_center = np.average(x_center_list)
      else:
        optimal_x_center = 0.5

      terminate_t = timeit.default_timer()
      await real_time_interpolate(pre_x_center, optimal_x_center, image_width, target_width, image, start_t)
      # terminate_t = timeit.default_timer()

      if (abs(pre_x_center - optimal_x_center) * image_width < 30): # 가로 기준(세로 고정) -> 세로 기준 추가해야 함
        optimal_x_center = pre_x_center
      
      left = int(optimal_x_center * image_width - target_width / 2)
      if (left < 0):
        left = 0
      elif (left > image_width - target_width):
        left = image_width - target_width

      pre_x_center = optimal_x_center
      img = image[:, left:left+target_width]

      # fps 계산
      # terminate_t = timeit.default_timer()
      fps = int(1.0 / (terminate_t - start_t))
      cv2.putText(img,
                  "FPS:" + str(fps),
                  (20, 60),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  2,
                  (0, 0, 255),
                  2)

      cv2.imshow('cropped', img)

      # cv2.imshow('image', image)

      if cv2.waitKey(10) & 0xFF == 27:
        break
  cap.release()
  cv2.destroyAllWindows()
  

if __name__ == "__main__":
  asyncio.run(main())