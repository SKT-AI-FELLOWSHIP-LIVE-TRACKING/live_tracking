import os
import cv2
import timeit
import tensorflow
import tensorflow as tf
import numpy as np
import math
from math import sqrt
from math import ceil
from dtos import anchor
from ssd_anchor_calculator import ssd_anchor_calculator
from dtos import DetectionLocaliztion


# 아래에 있는 함수 안에서 다 짠 다음에 class로 옮기려구 했는데...
# 안되면 다시 작성해야지.. ㅜㅜ
class object_detection():
	def __init__(self, image):
		self.image = image
		self.image_width = image.shape[1]
		self.image_height = image.shape[0]
		self.output_width = 320
		self.output_height = 320
		# self.interpreter
		# self.input_details
		# self.output_details
		self.num_coords = 4
		self.num_classes = 91
		self.num_boxes = 2034
		self.anchors = ssd_anchor_calculator().generate_ssd_anchor()
		self.labels = self.get_label("ssdlite_object_detection_labelmap.txt")
		self.detection_scores = [-1] * self.num_boxes # 각 박스에서의 score 최댓값
		self.detection_classes = [-1] * self.num_boxes # 각 박스에서 score 최댓값의 index 즉 class
		self.min_score_thresh = 0.5
		self.x_scale = 10
		self.y_scale = 10
		self.w_scale = 5
		self.h_scale = 5
		self.output_detections = []
	
	def image_preprocess(self):
		img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (self.output_width, self.output_height), fx=0, fy=0, interpolation=cv2.INTER_AREA) # (320, 320, 3)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 320, 320, 3)
		img = (img * (2.0 / 255.0) - 1).astype(np.float32) # (-1 ~ 1)

		return img
	
	def load_model(self):
		self.interpreter = tf.lite.Interpreter(model_path='ssdlite_object_detection.tflite')
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.interpreter.resize_tensor_input(self.input_details[0]['index'], [1, self.output_width, self.output_height, 3])
		self.interpreter.allocate_tensors()


	def detect_object(self, img):
		# input tensor
		self.interpreter.set_tensor(self.input_details[0]['index'], img)

		# run
		self.interpreter.invoke()

		# output tensor
		raw_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
		raw_boxes = raw_boxes.reshape(self.num_boxes, self.num_coords) # (2034, 4)

		raw_scores = self.interpreter.get_tensor(self.output_details[1]['index'])
		raw_scores = raw_scores.reshape(self.num_boxes, self.num_classes) # (2034, 91)

		# score tensor 처리
		for i in range(self.num_boxes):
			class_id = -1
			max_score = -float('inf') # 초기 값
			for score_idx in range(1, 91): # 0 class 무시
				score = raw_scores[i][score_idx]
				# 기존 코드에서는 스코어를 시그모이드로 한 번 더 계산
				# 근데 시그모이드 돌리는 순간 프레임 2~3 되버려서 일단 주석 처리
				# score = 1.0 / (1.0 + np.exp(-score))
				if (score > max_score):
					max_score = score
					class_id = score_idx
			self.detection_scores[i] = max_score
			self.detection_classes[i] = class_id
		
		boxes = self.decode_raw_boxes(raw_boxes, self.anchors)

		# detection에 저장
		for i in range(self.num_boxes):
			if (self.detection_scores[i] < self.min_score_thresh):
				continue
			# if(raw_boxes[i][2] < 0 or raw_boxes[i][3] < 0):
			# 	continue
			
			# detection에 localization 정보 담기
			# 
			print((boxes[i][0], 
								 				  boxes[i][1],
								 				  boxes[i][2],
								 				  boxes[i][3],
								 				  self.detection_scores[i],
								 				  self.detection_classes[i]))
			detection = self.convert_to_detection(boxes[i][0], 
								 				  boxes[i][1],
								 				  boxes[i][2],
								 				  boxes[i][3],
								 				  self.detection_scores[i],
								 				  self.detection_classes[i])
			self.output_detections.append(detection)

		return self.output_detections

	
	def get_label(self, label_file_path):
		with open(label_file_path) as f:
			lines = f.readlines()

		lines = [line.rstrip('\n') for line in lines]
		
		return lines
	
	def decode_raw_boxes(self, raw_boxes, anchors):
		boxes = [[-1 for i in range(4)] for n in range(2034)]

		for i in range(2034):
			y_center = raw_boxes[i][0]
			x_center = raw_boxes[i][1]
			h = raw_boxes[i][2]
			w = raw_boxes[i][3]

			# x_center = x_center / x_scale * anchors[i].w + anchors[i].x_center
			# y_center = y_center / y_scale * anchors[i].h + anchors[i].y_center
			# h = h / h_scale * anchors[i].h
			# w = w / w_scale * anchors[i].w

			y_center /= self.y_scale
			x_center /= self.x_scale
			h /= self.h_scale
			w /= self.w_scale


			y_center = y_center * anchors[i].h + anchors[i].y_center
			x_center = x_center * anchors[i].w + anchors[i].x_center
			h = math.exp(h) * anchors[i].h
			w = math.exp(w) * anchors[i].w
			

			ymin = y_center - h / 2.0
			xmin = x_center - w / 2.0
			ymax = y_center + h / 2.0
			xmax = x_center + w / 2.0

			# print(x_center, y_center)

			boxes[i][0] = xmin
			boxes[i][1] = ymin
			boxes[i][2] = xmax
			boxes[i][3] = ymax


		return boxes
	
	def convert_to_detection(xmin, ymin, xmax, ymax, score, class_id):
		detection = DetectionLocaliztion()

		detection.update_localization(xmin, ymin, xmax, ymax, score, class_id)

		return detection



# to_do_list
# 0. 프레임 연산 속도 어떻게..? - 현재 fps 약 16
# 1. label txt -> list로 얻어오기 (o) 
# 2. boxes 처리하기 (o)
# 3. nms

### 여기서부터 실험용 메인코드


from math import sqrt
from math import ceil
from dtos import anchor
from ssd_anchor_calculator import ssd_anchor_calculator

anchors = ssd_anchor_calculator().generate_ssd_anchor()


# raw box output tensor 처리하기
# tensor.shape -> (2034, 4)

def decode_raw_boxes(raw_boxes, anchors):
	boxes = [[-1 for i in range(4)] for n in range(2034)]

	x_scale = 10
	y_scale = 10
	w_scale = 5
	h_scale = 5

	for i in range(2034):
		y_center = raw_boxes[i][0]
		x_center = raw_boxes[i][1]
		h = raw_boxes[i][2]
		w = raw_boxes[i][3]

		# x_center = x_center / x_scale * anchors[i].w + anchors[i].x_center
		# y_center = y_center / y_scale * anchors[i].h + anchors[i].y_center
		# h = h / h_scale * anchors[i].h
		# w = w / w_scale * anchors[i].w

		y_center /= y_scale
		x_center /= x_scale
		h /= h_scale
		w /= w_scale


		y_center = y_center * anchors[i].h + anchors[i].y_center
		x_center = x_center * anchors[i].w + anchors[i].x_center
		h = math.exp(h) * anchors[i].h
		w = math.exp(w) * anchors[i].w
		

		ymin = y_center - h / 2.0
		xmin = x_center - w / 2.0
		ymax = y_center + h / 2.0
		xmax = x_center + w / 2.0

		# print(x_center, y_center)

		boxes[i][0] = xmin
		boxes[i][1] = ymin
		boxes[i][2] = xmax
		boxes[i][3] = ymax


	return boxes

# 처리한 좌표값들을 detection dataclass에 저장
def convert_to_detection(xmin, ymin, ymax, xmax, score, class_id):
	detection = DetectionLocaliztion()

	detection.update_localization(xmin, ymin, xmax, ymax, score, class_id)

	return detection


# 실험용 메인 코드
# 디텍션이 잘 되면 클래스로 옮겨서 코드 다시 작성할 예정
def detect_from_camera():
	min_score_thresh = 0.5
	# model 불러오기
	interpreter = tf.lite.Interpreter(model_path="ssdlite_object_detection.tflite")
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.resize_tensor_input(input_details[0]['index'], [1, 320, 320, 3])
	interpreter.allocate_tensors()
	

	cap = cv2.VideoCapture(0)
	while True:
		output_detections = []
		# 이미지 캡쳐
		ret, img_org = cap.read()
		start_t = timeit.default_timer()
		# print(img_org.shape)

		# 전처리
		img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (320, 320), fx=0, fy=0, interpolation=cv2.INTER_AREA)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 320, 320, 3)
		# img = (img / 255.).astype(np.float32) # (0 ~ 1)
		img = (img * (2.0 / 255.0) - 1).astype(np.float32) # (-1 ~ 1)

		# print(input_details)
		# print(output_details)

		# input tensor
		interpreter.set_tensor(input_details[0]['index'], img)

		# run
		interpreter.invoke()

		# output tensor
		raw_boxes = interpreter.get_tensor(output_details[0]['index'])

		raw_scores = interpreter.get_tensor(output_details[1]['index'])
		
		raw_boxes = raw_boxes.reshape(2034, 4)

		raw_scores = raw_scores.reshape(2034, 91)

		# 박스 갯수 = 2034
		detection_scores = [-1] * 2034 # 각 박스에서의 score 최댓값
		detection_classes = [-1] * 2034 # 각 박스에서 score 최댓값의 index 즉 class
		
		# score tensor 처리
		for i in range(raw_scores.shape[0]):
			class_id = -1
			max_score = -float('inf') # 초기 값
			for score_idx in range(1, 91): # 0 class 무시
				score = raw_scores[i][score_idx]
				# 기존 코드에서는 스코어를 시그모이드로 한 번 더 계산
				# 근데 시그모이드 돌리는 순간 프레임 2~3 되버려서 일단 주석 처리
				# score = 1.0 / (1.0 + np.exp(-score))
				if (score > max_score):
					max_score = score
					class_id = score_idx
			detection_scores[i] = max_score
			detection_classes[i] = class_id
		
		# box 값들 처리
		boxes = decode_raw_boxes(raw_boxes, anchors)

		# detection에 저장
		for i in range(2034):
			if (detection_scores[i] < min_score_thresh):
				continue
			# if(raw_boxes[i][2] < 0 or raw_boxes[i][3] < 0):
			# 	continue
			
			# detection에 localization 정보 담기
			# 
			detection = convert_to_detection(xmin=boxes[i][0], 
								 ymin=boxes[i][1],
								 xmax=boxes[i][2],
								 ymax=boxes[i][3],
								 score=detection_scores[i],
								 class_id=detection_classes[i])
			output_detections.append(detection)

		# 디텍션 이미지에 표시
		for i in range(len(output_detections)):
			detection = output_detections[i]

			# print(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
			ymin = int(detection.ymin * 720) # img_org.shape[0]
			xmin = int(detection.xmin * 1080) # img_org.shape[1]
			ymax = int(detection.ymax * 720) 
			xmax = int(detection.xmax * 1080)


			# print(xmin , ymin, xmax, ymax)
			cv2.rectangle(img_org, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
			
			cv2.putText(img_org,
				   str(detection.class_id),
				   (xmin, ymin),
				   cv2.FONT_HERSHEY_SIMPLEX,
				   5,
				   (0, 0, 255),
				   2)

		# fps 계산
		terminate_t = timeit.default_timer()
		fps = int(1.0 / (terminate_t - start_t))

		cv2.putText(img_org, "FPS:" + str(fps), (20,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 2)
		cv2.imshow('image', img_org)

		if cv2.waitKey(10) & 0xFF == 27:
			break
			
		

	cap.release()
	cv2.destroyAllWindows()

detect_from_camera()