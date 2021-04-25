from typing import List


class KeypointDTO(object):
    def __init__(self,
                 name: str,
                 score: float,
                 x: float,
                 y: float):
        self.name = name
        self.score = score
        self.x = x
        self.y = y


class FrameKeypointDTO:
    def __init__(self,
                 keypoints: List[KeypointDTO]):
        self.keypoints = keypoints


class VideoKeypointDTO:
    def __init__(self,
                 width: int,
                 height: int,
                 frames: List[FrameKeypointDTO]):
        self.width = width
        self.height = height
        self.frames = frames
