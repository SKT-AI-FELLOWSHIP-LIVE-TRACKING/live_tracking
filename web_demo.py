import cv2
import math
# import argparse
import timeit
import mediapipe as mp
from face_detection import face_detection
from mediapipe.framework.formats.detection_pb2 import Detection
from object_det_v2 import object_detection



def main():
  # For webcam input:
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
      
      print(image.shape)
      image_width = image.shape[1]
      image_height = image.shape[0]

      start_t = timeit.default_timer()

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      fd = face_detection(image)
      fd.detect_faces()
      regions = fd.localization_to_region()

      od = object_detection(image)

      output_dict, category_index = od.make_and_show_inference()

      
      if regions[0] != []:
        face_full_region = [regions[0][0].x, regions[0][0].y, regions[0][0].w, regions[0][0].h]

        x_px = min(math.floor(face_full_region[0] * image_width), image_width - 1)
        y_px = min(math.floor(face_full_region[1] * image_height), image_height - 1)
        w_px = min(math.floor(face_full_region[2] * image_width), image_width - 1)
        h_px = min(math.floor(face_full_region[3] * image_height), image_height - 1)
        cv2.rectangle(image, (x_px, y_px), (x_px+w_px, y_px+h_px), (0,0,255), 3)

        face_all_landmark = regions[2]
        for i in range(len(face_all_landmark)):
          all_landmark = face_all_landmark[i]
          for j in range(6):
            #print(all_landmark[j])
            x_px = min(math.floor(all_landmark[j].x * image.shape[1]), image.shape[1] - 1)
            y_px = min(math.floor(all_landmark[j].y * image.shape[0]), image.shape[0] - 1)
            w_px = int(all_landmark[j].w * image.shape[1])
            h_px = int(all_landmark[j].h * image.shape[0])
            #print(w_px,h_px)
            image = cv2.rectangle(image, (x_px, y_px), (x_px+w_px, y_px+h_px), (255,255,255), 3)

      boxes = output_dict['detection_boxes']
      classes = output_dict['detection_classes']
      scores = output_dict['detection_scores']

      for i in range(len(classes)):
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image_width, xmax * image_width,
                                    ymin * image_height, ymax * image_height)
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            print(left, right, top, bottom)
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            print(category_index[classes[i]]['name'])
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
      
      terminate_t = timeit.default_timer()
      fps = int(1.0 / (terminate_t - start_t))
      cv2.putText(image,
                  "FPS:" + str(fps),
                  (20, 60),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  2,
                  (0, 0, 255),
                  2)

      cv2.imshow('image', image)

        
      
      

      

      if cv2.waitKey(10) & 0xFF == 27:
        break
  cap.release()
  cv2.destroyAllWindows()
  

if __name__ == "__main__":
  main()