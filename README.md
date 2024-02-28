# Comfyui-Yolov8-JSON
This node is mainly based on the Yolov8 model for object detection, and it outputs related images, masks, and JSON information.

![image](https://github.com/Alysondao/Comfyui-Yolov8-JSON/blob/main/docs/workflow.png)

# README
- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# INSTALL
If you need to display JSON formatted data or save it, you need to install the [Comfyui-Toolbox](https://github.com/zcfrank1st/Comfyui-Toolbox) node in advance.

This node calls the official Python package, and you also need to install the following dependencies:

```
pip install ultralytics
```

# MODEL
This node supports automatic model downloads.
You can manually download the models to the specified folder `models/yolov8` from [Yolov8](https://github.com/ultralytics/ultralytics) as shown below:


```
ComfyUI
    models
        yolov8
            yolov8l.pt
            yolov8s.pt
            yolov8n.pt
            yolov8m.pt
            yolov8x.pt
```
 
