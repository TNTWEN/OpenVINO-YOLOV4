# OpenVINO-YOLOV4

## Introduction

 This is full implementation of YOLOV4,YOLOV4-relu,YOLOV4-tiny ,[YOLOV4-tiny-3l](https://github.com/TNTWEN/OpenVINO-YOLOV4/tree/v4-tiny-3l)in OpenVINO2020R4(or newer) .

 Based on https://github.com/mystic123/tensorflow-yolo-v3

## Latest Progress
- Pruned-OpenVINO-YOLO：https://github.com/TNTWEN/Pruned-OpenVINO-YOLO

  A tutorial on pruning the YOLOv3/v4 model(find the most compact model structure for the current detection task)and deploying it on OpenVINO which can even meet the simultaneous inference of multiple video streams. Both Chinese and English versions are available. Welcome to have a try!

- YOLOV4-tiny-3l:https://github.com/TNTWEN/OpenVINO-YOLOV4/tree/v4-tiny-3l      Welcome to have a try!

## FAQ 
[FAQ](https://github.com/TNTWEN/OpenVINO-YOLOV4/issues/10)

## Environment

- OpenVINO2020R4 :https://docs.openvinotoolkit.org/latest/index.html     or newer (please see FAQ Point 11)

- Win or Ubuntu

- Python 3.6.5

- Tensorflow 1.12.0 （1.15.4 for OpenVINO2021.1   ,   1.15.5 for OpenVINO2021.2 ）

- YOLOV4:https://github.com/AlexeyAB/darknet   train your own model

- *Convert YOLOV3/2/1 model :https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html


## How to use
★ This repository provides python inference demo for different OpenVINO version.[pythondemo](https://github.com/TNTWEN/OpenVINO-YOLOV4/tree/master/pythondemo)

★ Choose the right demo before you run object_detection_demo_yolov3_async.py

★ You could also use C++ inference demo provided by OpenVINO.
 
  (OpenVINO2021.2 default C++ demo path：`C:\Program Files (x86)\Intel\openvino_2021.2.185\inference_engine\demos\multi_channel\object_detection_demo_yolov3`)

### YOLOV4

prepare yolov4.weights .  

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC

"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python "C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU


```


 ![OpenVINOyolov4](assets/yolov4-416.png)

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




