import os
import cv2
import timeit
import tensorflow
import tensorflow as tf
import numpy as np
from dtos import DetectionLocaliztion
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(tf.config.list_physical_devices())

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# 아래에 있는 함수 안에서 다 짠 다음에 class로 옮기려구 했는데...
# 안되면 다시 작성해야지.. ㅜㅜ
class object_detection():
	def __init__(self, image):
		self.image = image
		self.image_width = image.shape[1]
		self.image_height = image.shape[0]
		self.output_width = 320
		self.output_height = 320
		self.interpreter
		self.input_details
		self.output_details
		self.box_tensor
		self.score_tensor
		self.labels
		self.detection_scores = []
		self.detection_classes = []
		self.min_score_thresh = 0.5
	
	def image_preprocess(self):
		img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (self.output_width, self.output_height))
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img = img.astype(np.float32)

		return img
	
	def load_model(self):
		self.interpreter = tf.lite.Interpreter(model_path='ssdlite_object_detection.tflite')
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()


	def detect_object(self, img):
		self.interpreter.set_tensor(self.input_details[0]['index'], img)
		self.interpreter.invoke()
		boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
		scores = self.interpreter.get_tensor(self.output_details[1]['index'])
		self.box_tensor = boxes[0]
		self.score_tensor = scores[0]
	
	def get_label(self, label_file_path):
		label_file_path = "ssdlite_object_detection_labelmap.txt"

		with open(label_file_path) as f:
			lines = f.readlines()

		lines = [line.rstrip('\n') for line in lines]
		print(lines)
		self.labels = lines

		

# to_do_list
# 0. 프레임 연산 속도 어떻게..? - 현재 fps 약 13~14
# 1. label txt -> list로 얻어오기 
# 2. boxes 처리하기 (***********)
# 3. 앵커 텐서 작성하기(?) 생략해도 될듯..?
# 4. (nms)

def convert_to_detection(y_center, x_center, h, w, score, class_id):
	detection = DetectionLocaliztion()
	# width = xmax - xmin
	# height = ymax - ymin
	# 이거 걍 막 해보는 중,,,
	xmin = x_center - w / 2
	ymin = y_center - h / 2
	xmax = w
	ymax = h


	detection.update_localization(ymin, xmin, ymax, xmax, score, class_id)

	return detection

def detect_from_camera():
	min_score_thresh = 0.5
	# load model
	interpreter = tf.lite.Interpreter(model_path="ssdlite_object_detection.tflite")
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.resize_tensor_input(input_details[0]['index'], [1, 320, 320, 3])
	interpreter.allocate_tensors()
	

	cap = cv2.VideoCapture(0)
	while True:
		start_t = timeit.default_timer()

		output_detections = []
		# capture image
		ret, img_org = cap.read()
		print(img_org.shape)

		# prepare input image
		img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (320, 320), fx=0, fy=0, interpolation=cv2.INTER_AREA)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 320, 320, 3)
		img = (img / 255.).astype(np.float32)

		# print(input_details)
		print(output_details)
		# with tf.device('/device:GPU:0'): 
		# # set input tensor
		interpreter.set_tensor(input_details[0]['index'], img)

		# run
		interpreter.invoke()

		# get output tensor
		raw_boxes = interpreter.get_tensor(output_details[0]['index'])
			
		# print(raw_boxes.shape)

		raw_scores = interpreter.get_tensor(output_details[1]['index'])
		
		raw_boxes = raw_boxes.reshape(2034, 4)

		raw_scores = raw_scores.reshape(2034, 91)
		# print(raw_scores.shape)
		# 박스 갯수 = 2034
		detection_scores = [-1] * 2034 # 각 박스에서의 score 최댓값
		detection_classes = [-1] * 2034 # 각 박스에서 score 최댓값의 index 즉 class
		

		for i in range(raw_scores.shape[0]):
			class_id = -1
			max_score = -float('inf') # 초기 값
			for score_idx in range(1, 91): # 0 class 무시
				score = raw_scores[i][score_idx]
				# 기존 코드에서는 스코어를 시그모이드로 한 번 더 계산하더라고??
				# 근데 시그모이드 돌리는 순간 프레임 2~3 되버려서 일단 주석해놨으
				# score = 1.0 / (1.0 + np.exp(-score))
				if (score > max_score):
					max_score = score
					class_id = score_idx
			detection_scores[i] = max_score
			detection_classes[i] = class_id

		for i in range(2034):
			if (detection_scores[i] < min_score_thresh):
				continue
			if(raw_boxes[i][2] < 0 or raw_boxes[i][3] < 0):
				continue
			# print(raw_boxes[i][0], 
			# 	  raw_boxes[i][1],
			# 	  raw_boxes[i][2],
			# 	  raw_boxes[i][3])
			# detection에 지역 정보 담을라고 했는데
			# box 처리가 잘 안되니까,,,, 일단 고민해보자
			detection = convert_to_detection(raw_boxes[i][0], 
								 raw_boxes[i][1],
								 raw_boxes[i][2],
								 raw_boxes[i][3],
								 detection_scores[i],
								 detection_classes[i])
			output_detections.append(detection)

		#print(detection_classes)
		#print(detection_scores)
		# s = max(detection_scores)
		# if(s > 0.5):
		# 	print("-------")
		# 	print(s)
		# 	idx = detection_scores.index(s)
		# 	print(detection_classes[idx])

		
		for i in range(len(output_detections)):
			detection = output_detections[i]
			# x = int(detection.x * img_org.shape[1])
			# y = int(detection.y * img_org.shape[0])
			# w = int(detection.w * img_org.shape[1])
			# h = int(detection.h * img_org.shape[0])
			# print("RAWS:", detection.x, detection.y, detection.w,detection.h)
			# print("DETECTIONS:", x, y, w, h)

			ymin = int(detection.ymin * 720)
			xmin = int(detection.xmin * 1080)
			ymax = int(detection.ymax * 720)
			xmax = int(detection.xmax * 1080)

			# 여기서부터
			# box = box.astype(np.int)
			cv2.rectangle(img_org, (ymin, xmin), (ymax, xmax), (255, 0, 0), 2)
			# cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
			cv2.putText(img_org,
				   str(detection.class_id),
				   (xmax, ymax),
				   cv2.FONT_HERSHEY_SIMPLEX,
				   10,
				   (0, 0, 255),
				   2)

		terminate_t = timeit.default_timer()
		fps = int(1.0 / (terminate_t - start_t))

		cv2.putText(img_org, "FPS:" + str(fps), (20,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 2)
		cv2.imshow('image', img_org)

		if cv2.waitKey(50) & 0xFF == 27:
			break
		
		
		# print("FPS:", fps)	
		

	cap.release()
	cv2.destroyAllWindows()

detect_from_camera()