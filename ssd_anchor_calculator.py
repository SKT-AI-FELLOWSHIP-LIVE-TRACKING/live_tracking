from math import sqrt
from math import ceil
from dtos import anchor
import cv2

class ssd_anchor_calculator:
    def __init__(self):
        self.num_layers = 6
        self.min_scale = 0.2
        self.max_scale = 0.95
        self.input_size_height = 320
        self.input_size_width = 320
        self.anchor_offset_x = 0.5
        self.anchor_offset_y = 0.5
        self.strides = [16, 32, 64, 128, 256, 512]
        self.strides_size = len(self.strides)
        self.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]
        self.aspect_ratios_size = len(self.aspect_ratios)
        self.interpolated_scale_aspect_ratio = 320.0 / 1080
        self.reduce_boxes_in_lowest_layer = True
        self.anchors = []
    
    def calculate_scale(self, min_scale, 
                    max_scale, 
                    stride_index, 
                    num_strides):
        if (num_strides == 1):
            return (min_scale + max_scale) * 0.5
        else:
            return (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0)
    
    def generate_ssd_anchor(self):
        layer_id = 0
        while (layer_id < self.num_layers):
            anchor_height = []
            anchor_width = []
            aspect_ratios_list = []
            scales = []

            last_same_stride_layer = layer_id
            while (last_same_stride_layer < self.strides_size and 
            self.strides[last_same_stride_layer] == self.strides[layer_id]):
                scale = self.calculate_scale(self.min_scale, self.max_scale, last_same_stride_layer, self.strides_size)

                if (last_same_stride_layer == 0 and self.reduce_boxes_in_lowest_layer):
                    aspect_ratios_list.append(1.0)
                    aspect_ratios_list.append(2.0)
                    aspect_ratios_list.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                else:
                    for i in range(self.aspect_ratios_size):
                        aspect_ratios_list.append(self.aspect_ratios[i])
                        scales.append(scale)
                    if(self.interpolated_scale_aspect_ratio > 0.0):
                        if (last_same_stride_layer == self.strides_size - 1):
                            scale_next = 1.0
                        else:
                            scale_next = self.calculate_scale(self.min_scale, self.max_scale, last_same_stride_layer + 1, self.strides_size)
                        scales.append(sqrt(scale * scale_next))
                        aspect_ratios_list.append(self.interpolated_scale_aspect_ratio) #interpolated
                last_same_stride_layer += 1
            

            for i in range(len(aspect_ratios_list)):
                ratio_sqrts  = sqrt(aspect_ratios_list[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)
            
            feature_map_height = 0
            feature_map_width = 0
            stride = self.strides[layer_id]
            feature_map_height = ceil(1.0 * self.input_size_height / stride)
            feature_map_width = ceil(1.0 * self.input_size_width / stride)

            # print(feature_map_width, feature_map_height, len(anchor_height))

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + self.anchor_offset_x) * 1.0 / feature_map_width
                        y_center = (y + self.anchor_offset_y) * 1.0 / feature_map_height

                        new_anchor = anchor()
                        new_anchor.set_x_center(x_center)
                        new_anchor.set_y_center(y_center)

                        new_anchor.set_w(anchor_width[anchor_id])
                        new_anchor.set_h(anchor_height[anchor_id])
                        self.anchors.append(new_anchor)
            layer_id = last_same_stride_layer

        return self.anchors





        

