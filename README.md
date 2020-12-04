# OpenVINO-YOLOV4

## Introduction

 This is full implementation of YOLOV4,YOLOV4-relu,YOLOV4-tiny in OpenVINO2020R4 .

 Based on https://github.com/mystic123/tensorflow-yolo-v3
 
## Latest Progress
Pruned-OpenVINO-YOLO：https://github.com/TNTWEN/Pruned-OpenVINO-YOLO

A tutorial on pruning the YOLOv3/v4 model(find the most compact model structure for the current detection task)and deploying it on OpenVINO which can even meet the simultaneous inference of Multiple video streams. Both Chinese and English versions are available. Welcome to have a try!

## Environment

OpenVINO2020R4 :https://docs.openvinotoolkit.org/latest/index.html

Win or Ubuntu

Tensorflow 1.12.0

YOLOV4:https://github.com/AlexeyAB/darknet   and download weights file

*Convert YOLOV3/2/1 model :https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html


## How to use

### YOLOV4

download yolov4.weights .  

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU


```


This is OpenVINO2020R4 object_detection_demo_yolov3_async.py without any change,but objects can still be detected normally


 ![OpenVINOyolov4](assets/yolov4-416.png)

CPU(intel i5-8250U)

Compared with darknet:
 ![darknetyolov4](assets/darknet-v4-416.jpg)

### YOLOV4-relu

download yolov4.weights .  

```
#windows  default OpenVINO path
cd yolov4-relu

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU
```



### YOLOV4-tiny

download yolov4-tiny.weights .  

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC --tiny

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolo_v4_tiny.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU
```

 ![OpenVINOyolov4tiny](assets/yolov4tiny416.png)

Compared with darknet:
 ![darknetyolov4tiny](assets/darknet-v4tiny-416.jpg)
### object_detection_demo_yolov4_async.py
(1)Add DIOU-NMS Support


