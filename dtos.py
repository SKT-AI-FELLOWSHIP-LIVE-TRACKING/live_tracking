from dataclasses import dataclass

@dataclass

#face_detection
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
