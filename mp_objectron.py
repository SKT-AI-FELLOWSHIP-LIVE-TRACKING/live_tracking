import cv2
import timeit
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.8,
                            model_name='Shoe') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    start_t = timeit.default_timer()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
    # Flip the image horizontally for a selfie-view display.

    terminate_t = timeit.default_timer()
    fps = int(1.0 / (terminate_t - start_t))

    cv2.putText(image, "FPS:" + str(fps), (20,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 2)
    cv2.imshow('MediaPipe Objectron', image)
    
    if (cv2.waitKey(5) & 0xFF == 27):
        break

cap.release()