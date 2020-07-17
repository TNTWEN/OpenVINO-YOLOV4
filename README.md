# OpenVINO-YOLOV4

## Introduction

 This is the implementation of YOLOV4(YOLOV4-RELU,YOLOV4-Tiny) in OpenVINO2020R4 .

I have verified the implementation of yolov4 and yolov4-relu.But there are still some problems in yolov4-tiny,i will try to fix it in the future.

## Environment

OpenVINO2020R4 :https://docs.openvinotoolkit.org/latest/index.html

Win or Ubuntu

YOLOV4:https://github.com/AlexeyAB/darknet   and download weights file

*Convert YOLOV3/2/1 model :https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html

## How to use

Openvino demo has scope restrictions.So in my code,i keep the YOLOV3 scope name,only rebuild YOLOV4 structure

### YOLOV4

download yolov4.weights .  and put it in yolov4 folder 

```
cd yolov4

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC

python yourfile/mo.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolov4.json --batch 1

cd ..

#remember to inintial openvino environment first !! 
python object_detection_demo_yolov3_async.py -i cam -m yolov4/frozen_darknet_yolov3_model.xml -d CPU


```

This is OpenVINO2020R4 object_detection_demo_yolov3_async.py without any change,but objects can still be detected normally

We still  need to update the "object_detection_demo_yolov3_async.py" to better adapt to the YOLOV4 in the future

 ![yolov4](assets/yolov4.png)

CPU(intel i5-8250U)

### YOLOV4-RELU

download yolov4.weights .  and put it in yolov4-relu folder 

```
cd yolov4-relu

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4.weights --data_format NHWC

python yourfile/mo.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolov4.json --batch 1
```



### YOLOV4-TINY

download yolov4-tiny.weights .  and put it in yolov4-tiny folder 

```
cd yolov4-tiny

python convert_weights_pb.py --class_names cfg/coco.names --weights_file yolov4-tiny.weights --data_format NHWC --tiny

python yourfile/mo.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolo_v4_tiny.json --batch 1
```

Existing problems:

Test on windows:

```
# convert .pb to IR.I'm not sure whether this warning will cause problems

warning：C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\model_optimizer\mo\middle\passes\fusing\decomposition.py:69: RuntimeWarning: invalid value encountered in sqrt
  scale = 1. / np.sqrt(variance.data.get_value() + eps)
```

```
# run the demo
--CPU: NAN error
--GPU: demo can run, but can't detect objects correctly,maybe need to update object_detection_demo_yolov3_async.py
```







```
_,split=tf.split(net,num_or_size_splits=2,axis=1 if data_format =="NCHW" else 3)
```

was not supported by OpenVINO yet.



so i use

```
    split =net[:,int(in_channels/2):,:,:]if data_format=="NCHW" else net[:,:,:,int(in_channels/2):]
```

to replace tf.split().  

You clould find it in yolo_v3_tiny.py .  I'm not sure whether that will lead to mistakes