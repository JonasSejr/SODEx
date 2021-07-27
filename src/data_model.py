from pathlib import Path

import cv2

class Image:
    def __init__(self, image_path):
        self.image_path: Path = image_path
        self.text_path: Path = image_path.parent / (image_path.stem + ".txt")
        self.size = None

    def as_array(self):
        image = cv2.imread(str(self.image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class PascalVOCObject:
    def __init__(self, bbox, image):
        self.image = image
        self.bbox = bbox  # (xmin, ymin, xmax, ymax, score, class)
