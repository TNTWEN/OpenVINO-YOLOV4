import os
import numpy as np
import argparse


def parse_model_cfg(path):
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    lines=[x for x in lines if x and not x.startswith("#")]
    lines=[x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer','stopbackward', 'cls_normalizer', 'iou_thresh','objectness_smooth','resize', 'obj_normalizer', 'new_coords', 'max_delta']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]
    assert not any(u), "Unsupported fields %s in %s." % (u, path)

    return mdefs



def generate(mdefs):
    yololayer=0
    for i,mdef in enumerate(mdefs):
        if mdef['type']=='shortcut':
            _from=mdef['from'][0]
            _fromid=i+_from
            mdefs[_fromid]['extra']="shortcut=inputs"

        if mdef['type']=='route':
            for x in mdef['layers']:
                if mdefs[i+x if x<0 else x]['type']=='maxpool':
                    pass
                else :
                    if 'extra' not in mdef.keys():
                        mdefs[i+x if x<0 else x]['extra']="route%s=inputs"%str(i+x if x<0 else x)
                        mdefs[i+x if x <0 else x]['route_name']="route%s"%str(i+x if x<0 else x)



        if mdef['type']=='maxpool':
            mdef['route_name']="maxpool%s"%str(i)
            mdef['extra']="maxpool%s=slim.max_pool2d(inputs, %s, 1, 'SAME')"%(str(i),mdef['size'])

        if mdef['type']=="yolo":
            yololayer+=1
            mdef['yoloid']=yololayer




    for i, mdef in enumerate(mdefs):
        if mdef['type']=="convolutional":
            if mdef['batch_normalize']:
                print("inputs = _conv2d_fixed_padding(inputs, %s, %s,strides=%s)"%(mdef['filters'],mdef['size'],mdef['stride']))
                if 'extra' in mdef.keys():
                    print(mdef['extra'])
        if mdef['type']=="shortcut":
            print("inputs = inputs + shortcut")
            if 'extra'in mdef.keys():
                print(mdef['extra'])
        if mdef['type']=="maxpool":
            print(mdef['extra'])
        if mdef['type']=='route':
            if len(mdef['layers'])==1:
                print("inputs=%s"%mdefs[i+mdef["layers"][0] if mdef["layers"][0]<0 else mdef["layers"][0]]['route_name'])
            if len(mdef['layers'])==4:
                print("inputs=tf.concat([%s,%s,%s,%s],axis=1 if data_format == 'NCHW' else 3)"%(mdefs[i+mdef["layers"][0] if mdef["layers"][0]<0 else mdef["layers"][0]]['route_name'],\
                                                       mdefs[i+mdef["layers"][1] if mdef["layers"][1]<0 else mdef["layers"][1]]['route_name'],\
                                                       mdefs[i+mdef["layers"][2] if mdef["layers"][2]<0 else mdef["layers"][2]]['route_name'],\
                                                       mdefs[i+mdef["layers"][3] if mdef["layers"][3]<0 else mdef["layers"][3]]['route_name']))
            if len(mdef['layers'])==2:
                print("inputs=tf.concat([%s,%s],axis=1 if data_format == 'NCHW' else 3)"%(mdefs[i+mdef["layers"][0] if mdef["layers"][0]<0 else mdef["layers"][0]]['route_name'],\
                                                 mdefs[i+mdef["layers"][1] if mdef["layers"][1]<0 else mdef["layers"][1]]['route_name']))

        if mdef['type']=="upsample":
            print("inputs = _upsample(inputs,route%s.get_shape().as_list(),data_format)"%mdefs[i+1]["layers"][0])
            if 'extra'in mdef.keys():
                print(mdef['extra'])
        if mdef['type']=="yolo":
            print("detect_%s = _detection_layer(inputs, num_classes, _ANCHORS[%s:%s], img_size, data_format)\ndetect_%s = tf.identity(detect_%s, name='detect_%s')"%(mdef['yoloid'],str(mdef['mask'][0]),str(mdef['mask'][2]+1),mdef['yoloid'],mdef['yoloid'],mdef['yoloid']))







if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",type=str,default='cfg/yolov3.cfg',help=('*.cfg path'))
    opt = parser.parse_args()
    path = opt.cfg
    mdefs=parse_model_cfg(path)
    mdefs.pop(0)
    generate(mdefs)
    print("detections = tf.concat([detect_1, detect_2, detect_3], axis=1)")
    print("detections = tf.identity(detections, name='detections')")
