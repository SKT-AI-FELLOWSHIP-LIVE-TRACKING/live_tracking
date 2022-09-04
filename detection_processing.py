from dtos import DetectionRegions


class detection_processing():
    def __init__(self, boxes, classes, scores, face_regions):
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.face_regions = face_regions
        self.categories = [0, # person
                           15,16,17,18,19,20,21,22,23,24, # 동물
                           36] # sports ball
        self.regions = []
        # self.regions_to_sort
        # self.sorted_regions

    def tensors_to_regions(self):
        for i, box in enumerate(self.boxes):
            if (self.scores[i] < 0.6):
                continue
            x = box[1]
            y = box[0]
            w = box[3] - box[1]
            h = box[2] - box[0]
            detection = DetectionRegions(x, y, w, h, self.scores[i], self.classes[i])
            self.regions.append(detection)
        # self.remove_unrequired_categories(self.regions)
        # self.combine_faces_and_detections(self.regions)
        required_regions = self.remove_unrequired_categories(self.regions)
        self.combine_faces_and_detections(required_regions)
        
    def remove_unrequired_categories(self, detection_regions):
        for i, region in enumerate(detection_regions):
            if (self.binary_search(self.categories, region.class_id) == False):
                detection_regions.pop(i)
        # self.regions = detection_regions
        return detection_regions

    def combine_faces_and_detections(self, detection_regions):
        # self.face_regions
        # detection_regions
        idx_ = 0
        pop_flag = 0
        while (idx_ != len(detection_regions)):
            region = detection_regions[idx_]
            if (region.class_id == 0):
                for j, face in enumerate(self.face_regions):
                    if ((face.x > region.x) and (face.x + face.w < region.x + region.w)):
                        # if (face.score < 0.7):
                        #     self.face_regions.pop(j)
                        # else:
                        #     detection_regions.pop(i)
                        #     break

                        if(region.score > face.score):
                            self.face_regions.pop(j)
                        else:
                            detection_regions.pop(idx_)
                            pop_flag = 1
                            break
            if (pop_flag == 1):
                pop_flag = 0
            else:
                idx_ += 1
          

        regions_to_sort = self.face_regions + detection_regions
        # print(regions_to_sort)
        self.regions_to_sort = regions_to_sort

    
    def sort_detection(self):
        # regions_to_sort = self.faceregions + self.regions
        self.sorted_regions = self.insert_sort(self.regions_to_sort)

        return self.sorted_regions
        

    def insert_sort(self, array):
        # print(array)
        for i in range(1, len(array)):
            for j in range(i, 0, -1):
                if array[j].score > array[j - 1].score: 
                    array[j], array[j - 1] = array[j - 1], array[j]
                else: 
                    break
        return array

    def binary_search(self, array, search):
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
            return self.binary_search(array[median:], search)
        else:
            return self.binary_search(array[:median], search)

