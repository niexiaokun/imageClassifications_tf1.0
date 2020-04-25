import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
    return x


def conv_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=tf.nn.relu)
        # normalizer_fn is None by default
    return x


def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
    return x


def conv_bn(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=None)
    return x


def dw_conv_bn(x, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'dw_conv_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None)
    return x


def dw_conv_bn_relu(x, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'dw_conv_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
    return x


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(
        flops.total_float_ops/1024/1024/1024, params.total_parameters/1024/1024))
