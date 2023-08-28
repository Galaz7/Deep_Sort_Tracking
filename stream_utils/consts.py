from enum import Enum

class ResultsSendMethod(Enum):
    NO_SEND = 0
    FRAMES_WITH_DETECTIONS = 1
    ONLY_DETECTIONS = 2
