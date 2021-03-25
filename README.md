# OpenVINO-YOLOV4-tiny-3l

[cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-3l.cfg)

## Environment

OpenVINO2021.3 :https://docs.openvinotoolkit.org/latest/index.html    (After OpenVINO2020R4)

Inference device: Intel CPU/GPU/VPU/NCS2

Inference demo: python and C++

Win or Ubuntu

Tensorflow 1.15.5

Training：https://github.com/AlexeyAB/darknet




## How to use

Train your yolov4-tiny-3l model first

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny-3l.weights --data_format NHWC

"C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"

python "C:\Program Files (x86)\Intel\openvino_2021.3.394\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU

```

Tips:

1. python demo for different OpenVINO version:https://github.com/TNTWEN/OpenVINO-YOLOV4/tree/master/pythondemo

2. Compile C++ demo by yourself:(OpenVINO2021.3 default C++ demo path：`C:\Program Files (x86)\Intel\openvino_2021.3.394\deployment_tools\open_model_zoo\demos\multi_channel_object_detection_demo_yolov3\cpp`)

3. How to use custom model:

   (1)  When running convert_weights_pb.py use your .names file

   (2)  Modify "classes" in yolov4.json

