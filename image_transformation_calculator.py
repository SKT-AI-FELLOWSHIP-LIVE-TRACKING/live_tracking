import cv2
import numpy as np

class image_transformation_calculator:
    def __init__(self, image):
        self.image = image
        self.input_width = image.shape[1]
        self.input_height = image.shape[0]
        self.output_width = 320
        self.output_height = 320
        
        # self.Resize()

    
    def Resize(self):
        scale = min(float(self.output_width) / self.input_width,  float(self.output_height) / self.input_height)
        target_width = round(self.input_width * scale)
        target_height = round(self.input_height * scale)
        #print(target_width, target_height)

        if scale < 1.0:
            scale_flag = cv2.INTER_AREA
        else:
            scale_flag = cv2.INTER_LINEAR

        
        dst = cv2.resize(self.image, dsize=(320, 320), interpolation=scale_flag)

        return dst
