# OpenVINO-YOLOV4

## Introduction

 This is the implementation of YOLOV4,YOLOV4-relu,YOLOV4-tiny in OpenVINO2020R4 .

This update has replaced all v3 interfaces name with v4.  In addition to  the JSON file:  "id": "TFYOLOV3"(demo code needs to be modified if  "id": "TFYOLOV4  is used) .we want to make sure that you can run the demo on your PC directly



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

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU


```


This is OpenVINO2020R4 object_detection_demo_yolov3_async.py without any change,but objects can still be detected normally


 ![OpenVINOyolov4](assets/yolov4-416.png)

CPU(intel i5-8250U)

Compared with darknet:
 ![darknetyolov4](assets/darknet-v4-416.png)

### YOLOV4-relu

download yolov4.weights .  

```
#windows  default OpenVINO path
cd yolov4-relu

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU
```



### YOLOV4-tiny

download yolov4-tiny.weights .  

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC --tiny

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolo_v4_tiny.json --batch 1

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU
```

 ![OpenVINOyolov4tiny](assets/yolov4tiny416.png)
 
Compared with darknet:
 ![darknetyolov4tiny](assets/darknet-v4tiny-416.png)
### object_detection_demo_yolov4_async.py
(1)Add DIOU-NMS Support
