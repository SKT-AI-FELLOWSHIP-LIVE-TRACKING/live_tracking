from dtos import DetectionRegions

class detection_processing():
    def __init__(self, boxes, classes, scores, faceregions):
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.faceregions = faceregions
        self.regions = []
        self.sorted_regions = []

    def tensors_to_regions(self):
        # 얼굴 수와 사람의 디텍션 수가 같을 때는 프레임 처리를 위해 사람의 디텍션 정보 추가 ㅌ
        ignore_person = 0
        person_cnt = len([x for x in self.classes if x == 0])
        if (len(self.faceregions) == person_cnt):
            ignore_person = 1
        for i, box in enumerate(self.boxes):
            if (ignore_person == 1 and self.classes[i] == 0):
                continue
            x = box[1]
            y = box[0]
            w = box[3] - box[1]
            h = box[2] - box[0]
            detection = DetectionRegions(x, y, w, h, self.scores[i], self.classes[i])
            self.regions.append(detection)
    
    def sort_detection(self):
        regions_to_sort = self.faceregions + self.regions
        self.sorted_regions = self.insert_sort(regions_to_sort)

        return self.sorted_regions
        

    def insert_sort(self, array):
        print(array)
        for i in range(1, len(array)):
            for j in range(i, 0, -1):
                if array[j].score > array[j - 1].score: 
                    array[j], array[j - 1] = array[j - 1], array[j]
                else: 
                    break
        return array
