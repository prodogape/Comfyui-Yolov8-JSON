from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "Load Yolov8 Model": LoadYolov8Model,
    "Load Yolov8 Model From Path": LoadYolov8ModelFromPath,
    "Apply Yolov8 Model": ApplyYolov8Model,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Yolov8 Model": "Load Yolov8 Model",
    "Load Yolov8 Model Upload": "Load Yolov8 Model From Path",
    "Apply Yolov8 Model": "Apply Yolov8 Model",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS" ,"WEB_DIRECTORY"]
