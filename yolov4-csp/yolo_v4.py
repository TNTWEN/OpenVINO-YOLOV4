
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(12, 16), (19, 36), (40, 28),
            (36, 75), (76, 55), (72, 146),
            (142, 110), (192, 243), (459, 401)]
@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]],
                               mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs



def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _yolo_res_Block(inputs,in_channels,res_num,data_format,double_ch=False):
    out_channels = in_channels
    if double_ch:
        out_channels = in_channels * 2
    net = _conv2d_fixed_padding(inputs,in_channels*2,kernel_size=3,strides=2)
    route = _conv2d_fixed_padding(net,out_channels,kernel_size=1)
    net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)

    for _ in range(res_num):
        tmp=net
        net = _conv2d_fixed_padding(net,in_channels,kernel_size=1)
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=3)
        #shortcut
        net = tmp+net

    net=_conv2d_fixed_padding(net,out_channels,kernel_size=1)

    #concat
    net=tf.concat([net,route],axis=1 if data_format == 'NCHW' else 3)
    net=_conv2d_fixed_padding(net,in_channels*2,kernel_size=1)
    return net

def _yolo_conv_block(net,in_channels,a,b):
    for _ in range(a):
        out_channels=in_channels/2
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)
        net = _conv2d_fixed_padding(net,in_channels,kernel_size=3)

    out_channels=in_channels
    for _ in range(b):
        out_channels=out_channels/2
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)

    return net


def _spp_block(inputs, data_format='NCHW'):
    return tf.concat([slim.max_pool2d(inputs, 13, 1, 'SAME'),
                      slim.max_pool2d(inputs, 9, 1, 'SAME'),
                      slim.max_pool2d(inputs, 5, 1, 'SAME'),
                      inputs],
                     axis=1 if data_format == 'NCHW' else 3)


def _upsample(inputs, out_shape, data_format='NCHW'):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        new_height = out_shape[2]
        new_width = out_shape[3]
    else:
        new_height = out_shape[1]
        new_width = out_shape[2]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs



def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                              stride=1, normalizer_fn=None,
                              activation_fn=None,
                              biases_initializer=tf.zeros_initializer())
    #predictions = 1/1+tf.exp(-predictions)
    print("**********************************************************")
    print(predictions.name)
    print("**********************************************************")
    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    if data_format == 'NCHW':
        predictions = tf.reshape(
            predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride
    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions




def yolo_v4(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v4 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :param with_spp: whether or not is using spp layer.
    :return:
    """

    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x:x* tf.math.tanh(tf.math.softplus(x))):
            with tf.variable_scope('yolo-v4'):
                inputs = _conv2d_fixed_padding(inputs, 32, 3,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 64, 3,strides=2)
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 32, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 64, 3,strides=1)
                inputs = inputs + shortcut
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=2)
                route5=inputs
                inputs = _conv2d_fixed_padding(inputs, 64, 1,strides=1)
                route6=inputs
                inputs=route5
                inputs = _conv2d_fixed_padding(inputs, 64, 1,strides=1)
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 64, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 64, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 64, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 64, 3,strides=1)
                inputs = inputs + shortcut
                inputs = _conv2d_fixed_padding(inputs, 64, 1,strides=1)
                route15=inputs
                inputs=tf.concat([route15,route6],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=2)
                route18=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                route19=inputs
                inputs=route18
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = inputs + shortcut
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                route46=inputs
                inputs=tf.concat([route46,route19],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route48=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=2)
                route49=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route50=inputs
                inputs=route49
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = inputs + shortcut
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route77=inputs
                inputs=tf.concat([route77,route50],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route79=inputs
                inputs = _conv2d_fixed_padding(inputs, 1024, 3,strides=2)
                route80=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route81=inputs
                inputs=route80
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                inputs = inputs + shortcut
                shortcut=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                inputs = inputs + shortcut
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route96=inputs
                inputs=tf.concat([route96,route81],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 1024, 1,strides=1)
                route98=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route99=inputs
                inputs=route98
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route103=inputs
                maxpool104=slim.max_pool2d(inputs, 5, 1, 'SAME')
                inputs=route103
                maxpool106=slim.max_pool2d(inputs, 9, 1, 'SAME')
                inputs=route103
                maxpool108=slim.max_pool2d(inputs, 13, 1, 'SAME')
                inputs=tf.concat([maxpool108,maxpool106,maxpool104,route103],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                route111=inputs
                inputs=tf.concat([route111,route99],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route113=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _upsample(inputs,route79.get_shape().as_list(),data_format)
                route115=inputs
                inputs=route79
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route117=inputs
                inputs=tf.concat([route117,route115],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route119=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route120=inputs
                inputs=route119
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                route125=inputs
                inputs=tf.concat([route125,route120],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route127=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _upsample(inputs,route48.get_shape().as_list(),data_format)
                route129=inputs
                inputs=route48
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                route131=inputs
                inputs=tf.concat([route131,route129],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                route133=inputs
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                route134=inputs
                inputs=route133
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 128, 3,strides=1)
                route139=inputs
                inputs=tf.concat([route139,route134],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 128, 1,strides=1)
                route141=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')
                inputs=route141
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=2)
                route146=inputs
                inputs=tf.concat([route146,route127],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route148=inputs
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route149=inputs
                inputs=route148
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 256, 3,strides=1)
                route154=inputs
                inputs=tf.concat([route154,route149],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 256, 1,strides=1)
                route156=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')
                inputs=route156
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=2)
                route161=inputs
                inputs=tf.concat([route161,route113],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route163=inputs
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                route164=inputs
                inputs=route163
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 512, 3,strides=1)
                route169=inputs
                inputs=tf.concat([route169,route164],axis=1 if data_format == 'NCHW' else 3)
                inputs = _conv2d_fixed_padding(inputs, 512, 1,strides=1)
                inputs = _conv2d_fixed_padding(inputs, 1024, 3,strides=1)
                detect_3 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')
                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')

                return detections

