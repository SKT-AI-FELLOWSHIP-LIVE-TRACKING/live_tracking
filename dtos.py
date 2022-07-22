from dataclasses import dataclass

@dataclass

# face_detection
class FaceRegions:
    x: float
    y: float
    w: float
    h: float
    SignalType: str
    score: float

    def update_region(self, x, y, w, h, SignalType, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.SignalType = SignalType
        self.score = score
    
    def set_x(self, x):
        self.x = x
    
    def set_y(self, y):
        self.y = y

    def set_w(self, w):
        self.w = w

    def set_h(self, h):
        self.h = h


# object detection
class DetectionLocaliztion:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    class_id: int
    format: str

    def update_localization(self, xmin, ymin, xmax, ymax, score, class_id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.class_id = class_id
    
    def set_x(self, x):
        self.x = x
    
    def set_y(self, y):
        self.y = y

    def set_w(self, w):
        self.w = w

    def set_h(self, h):
        self.h = h

# anchor
class anchor:
    x_center: float
    y_center: float
    w: float
    h: float
    
    def set_x_center(self, x_center):
        self.x_center = x_center
    
    def set_y_center(self, y_center):
        self.y_center = y_center

    def set_w(self, w):
        self.w = w

    def set_h(self, h):
        self.h = h


