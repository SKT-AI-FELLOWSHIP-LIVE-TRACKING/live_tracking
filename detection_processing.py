from dtos import DetectionRegions

class detection_processing():
    def __init__(self, boxes, classes, scores, face_regions):
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.face_regions = face_regions
        self.regions = []
        # self.regions_to_sort
        # self.sorted_regions

    def tensors_to_regions(self):
        for i, box in enumerate(self.boxes):
            x = box[1]
            y = box[0]
            w = box[3] - box[1]
            h = box[2] - box[0]
            detection = DetectionRegions(x, y, w, h, self.scores[i], self.classes[i])
            self.regions.append(detection)
        self.combine_faces_and_detections(self.regions)
        
    def combine_faces_and_detections(self, detectioin_regions):
        # self.face_regions
        # detectioin_regions
        for i, region in enumerate(detectioin_regions):
            if (region.class_id == 0):
                for j, face in enumerate(self.face_regions):
                    if ((face.x > region.x) and (face.x + face.w < region.x + region.w)):
                        if(region.score > face.score):
                            self.face_regions.pop(j)
                        else:
                            detectioin_regions.pop(i)
                            break
        regions_to_sort = self.face_regions + detectioin_regions
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
