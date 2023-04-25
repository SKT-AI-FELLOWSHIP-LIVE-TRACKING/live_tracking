import numpy as np
import tensorflow as tf
import cv2
import timeit

class object_detection():
    def __init__(self, image):
        """
        image : Array of uint8
            Raw image to find predictions
        interpreter : tensorflow.lite.python.interpreter.Interpreter
            tflite model interpreter
        input_details : list
            input details of interpreter
        output_details : list
            output details of interpreter
        category_index : dict
            dictionary of labels
        """
        self.image = image
        self.interpreter = tf.lite.Interpreter(model_path="coco_ssd_mobilenet/detect.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.category_index = self.create_category_index('coco_ssd_mobilenet/labelmap.txt')
        self.input_shape = self.input_details[0]['shape']
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]

    def create_category_index(self, label_path='coco_ssd_mobilenet/labelmap.txt'):
        """
        To create dictionary of label map
        -------
        Parameters:
            label_path : string, optional
                Path to labelmap.txt. The default is 'coco_ssd_mobilenet/labelmap.txt'.
        -------
        Returns:
            category_index : dict
                dictionary of labels.
        """
        f = open(label_path)
        category_index = {}
        for i, val in enumerate(f):
            if i != 0:
                val = val[:-1]
                if val != '???':
                    category_index.update({(i-1): {'id': (i-1), 'name': val}})
                
        f.close()
        return category_index

    def get_output_dict(self, image, nms=True, iou_thresh=0.5, score_thresh=0.7):
        """
        Function to make predictions and generate dictionary of output
        --------
        Parameters:
            image : Array of uint8
                Preprocessed Image to perform prediction on
            nms : bool, optional
                To perform non-maximum suppression or not. The default is True.
            iou_thresh : float, optional
                Intersection Over Union Threshold. The default is 0.5.
            score_thresh : float, optional
                score above predicted class is accepted. The default is 0.6.
        -------
        Returns:
            output_dict : dict
                Dictionary containing bounding boxes, classes and scores.
        """
        output_dict = {
                    'detection_boxes' : self.interpreter.get_tensor(self.output_details[0]['index'])[0],
                    'detection_classes' : self.interpreter.get_tensor(self.output_details[1]['index'])[0],
                    'detection_scores' : self.interpreter.get_tensor(self.output_details[2]['index'])[0],
                    'num_detections' : self.interpreter.get_tensor(self.output_details[3]['index'])[0]
                    }

        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        if nms:
            output_dict = self.apply_nms(output_dict, iou_thresh, score_thresh)
        return output_dict

    def apply_nms(self, output_dict, iou_thresh=0.5, score_thresh=0.6):
        """
        Function to apply non-maximum suppression on different classes
        ----------
        Parameters
            output_dict : dictionary
                dictionary containing:
                    'detection_boxes' : Bounding boxes coordinates. Shape (N, 4)
                    'detection_classes' : Class indices detected. Shape (N)
                    'detection_scores' : Shape (N)
                    'num_detections' : Total number of detections i.e. N. Shape (1)
            iou_thresh : float, optional
                Intersection Over Union threshold value. The default is 0.5.
            score_thresh : float, optional
                Score threshold value below which to ignore. The default is 0.6.
        -------
        Returns
            output_dict : dictionary
                dictionary containing only scores and IOU greater than threshold.
                    'detection_boxes' : Bounding boxes coordinates. Shape (N2, 4)
                    'detection_classes' : Class indices detected. Shape (N2)
                    'detection_scores' : Shape (N2)
                    N2 is the number of valid predictions after those conditions.
        """
        q = 90 # no of classes
        num = int(output_dict['num_detections'])
        boxes = np.zeros([1, num, q, 4])
        scores = np.zeros([1, num, q])
        # val = [0]*q
        for i in range(num):
            # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
            boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
            scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
        nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                    scores=scores,
                                                    max_output_size_per_class=num,
                                                    max_total_size=num,
                                                    iou_threshold=iou_thresh,
                                                    score_threshold=score_thresh,
                                                    pad_per_class=False,
                                                    clip_boxes=False)
        valid = nmsd.valid_detections[0].numpy()
        output_dict = {
                    'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                    'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                    'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                    }
        return output_dict

    def visualize_image(self, img, boxes, classes, scores, category_index):
        for i in range(len(classes)):
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * self.image_width, xmax * self.image_width,
                                    ymin * self.image_height, ymax * self.image_height)
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)

            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img,
                    str(category_index[classes[i]]['name']),
                    (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    2)
            cv2.putText(img,
                    str(scores[i]),
                    (left, top + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    2)

            cv2.imshow('image', img)



    def detect_objects(self, nms=True, score_thresh=0.6, iou_thresh=0.5):
        """
        Function to detect objects
        -------
        Parameters:
            img : Array of uint8
                Original Image to find predictions on.
            
            nms : bool, optional
                To perform non-maximum suppression or not. The default is True.
            score_thresh : int, optional
                score above predicted class is accepted. The default is 0.6.
            iou_thresh : int, optional
                Intersection Over Union Threshold. The default is 0.5.
        -------
        Returns
            output_dict : dictionary
                        'detection_boxes' : Bounding boxes coordinates. Shape (N2, 4)
                        'detection_classes' : Class indices detected. Shape (N2)
                        'detection_scores' : Shape (N2)
                        N2 is the number of valid predictions after those conditions.
            category_index : dict
                    dictionary of labels.
        """
        img = self.image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
        img_rgb = img_rgb.reshape([1, 300, 300, 3])

        self.interpreter.set_tensor(self.input_details[0]['index'], img_rgb)
        self.interpreter.invoke()
        
        output_dict = self.get_output_dict(img_rgb, nms, iou_thresh, score_thresh)
        # Visualization of the results of a detection.
        # self.visualize_image(img, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], self.category_index)


        return output_dict, self.category_index
        