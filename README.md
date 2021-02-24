# OpenVINO-YOLOV4-tiny-3l

[cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-3l.cfg)

## Environment

OpenVINO2021.1 :https://docs.openvinotoolkit.org/latest/index.html    (OpenVINO2020R4 or newer)

Inference device: Intel CPU/GPU/VPU/NCS2

Inference demo: python and C++

Win or Ubuntu

Tensorflow 1.15.4

Training：https://github.com/AlexeyAB/darknet




## How to use

Train your yolov4-tiny-3l model first

```
#windows  default OpenVINO path

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny-3l.weights --data_format NHWC

"C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"

python "C:\Program Files (x86)\Intel\openvino_2021.1.110\deployment_tools\model_optimizer\mo.py" --input_model frozen_darknet_yolov4_model.pb --transformations_config yolov4.json --batch 1 --reverse_input_channels

python object_detection_demo_yolov3_async.py -i cam -m frozen_darknet_yolov4_model.xml  -d CPU

```

Tips:

1. If you use other OpenVINO version,please use python or C++ inference demo provided by the OpenVINO you download!

2. How to use custom model:

   (1)  When running convert_weights_pb.py use your .names file

   (2)  Modify "classes" in yolov4.json

