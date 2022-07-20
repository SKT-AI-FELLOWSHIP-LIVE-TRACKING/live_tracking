import cv2
import math
# import argparse
import mediapipe as mp
from face_detection import face_detection
from image_transformation_calculator import image_transformation_calculator
from mediapipe.framework.formats.detection_pb2 import Detection

def main():
  # For webcam input:
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      cal = image_transformation_calculator(image)
      cropped_frame = cal.Resize()

      cv2.imshow('WebCam', cropped_frame)

      # fd = face_detection(image)
      # fd.DetectFaces()
      # regions = fd.LocalizationToRegion()

      
      # if regions[0] != []:
      #   face_full_region = [regions[0][0].x, regions[0][0].y, regions[0][0].w, regions[0][0].h]

      #   x_px = min(math.floor(face_full_region[0] * image.shape[1]), image.shape[1] - 1)
      #   y_px = min(math.floor(face_full_region[1] * image.shape[0]), image.shape[0] - 1)
      #   w_px = min(math.floor(face_full_region[2] * image.shape[1]), image.shape[1] - 1)
      #   h_px = min(math.floor(face_full_region[3] * image.shape[0]), image.shape[0] - 1)
      #   # cropped_frame = image[y_px:y_px+h_px,x_px:x_px+w_px]

      #   face_all_landmark = regions[2]
      #   for i in range(len(face_all_landmark)):
      #     all_landmark = face_all_landmark[i]
      #     for j in range(6):
      #       #print(all_landmark[j])
      #       x_px = min(math.floor(all_landmark[j].x * image.shape[1]), image.shape[1] - 1)
      #       y_px = min(math.floor(all_landmark[j].y * image.shape[0]), image.shape[0] - 1)
      #       w_px = int(all_landmark[j].w * image.shape[1])
      #       h_px = int(all_landmark[j].h * image.shape[0])
      #       #print(w_px,h_px)
      #       image = cv2.rectangle(image, (x_px, y_px), (x_px+w_px, y_px+h_px), (0,0,255), 3)

      #   cv2.imshow('WebCam', cv2.flip(image, 1))
        # cv2.imshow('WebCam', cv2.flip(cropped_frame, 1))
      
      # else:
      #   cv2.imshow('WebCam', cv2.flip(image, 1))

      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
  cv2.destroyAllWindows()
  

if __name__ == "__main__":
  main()