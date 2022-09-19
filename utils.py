from logging import raiseExceptions
import cv2
import timeit
import numpy as np
import math



def get_ratio(image_width, image_height):
  return image_width / image_height

def round_to_even(value):
  rounded_value = round(value)

  if (rounded_value % 2 == 1):
    rounded_value = max(2, rounded_value - 1)
  
  return rounded_value

# cosine similarity 
def cos_sim(A, B):
  return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))


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
    scaled_target_height = target_height / image_height / 2

    return target_width, target_height, scaled_target_height


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

class piecewise_func():
  def __init__(self, start, end, time):
    self.start_x = 0
    self.start_y = start
    self.end_x = time
    self.end_y = end
    # self.time_ = 30 # fps 30 -> 1 sec
  
  def evaluate(self, input):
    return self.end_x - (self.end_x - input) / (self.end_x - self.start_x) * (self.end_y - self.start_y)

# test
# 카메라 기준 1초
async def real_time_interpolate(pre_x_center, optimal_x_center, image_width, target_width, image, start_t):
  time_ = 30 # fps 30
  start = pre_x_center
  end = optimal_x_center
  func = piecewise_func(start, end, time_)
  for i in range(1, time_):
    interpolated = func.evaluate(i)
    if (int(interpolated * image_width) == optimal_x_center):
      break
    float_frame_imshow(interpolated, image_width, target_width, image, start_t)



### image - RGB
def get_features(bbox_tlbr, image_rgb, feature_extractor):
    im_crops = []

    for box in bbox_tlbr:
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        im = image_rgb[y1:y2, x1:x2]
        im_crops.append(im)
    if im_crops:
        features = feature_extractor(im_crops)
    else:
        features = np.array([])
    return features