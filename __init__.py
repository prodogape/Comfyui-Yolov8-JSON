from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "Load Yolov8 Model": LoadYolov8Model,
    "Apply Yolov8 Model One Label": ApplyYolov8ModelOneLabel,
    "Apply Yolov8 Model": ApplyYolov8Model,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
