from cnn_net.conv_wraper import *
from collections import OrderedDict
import time


def mobilenet1_unit(x, out_channel, kernel_size, stride, dilation=1, name='mobilenet1_unit', reuse=None):
    with tf.variable_scope(None, name, reuse=reuse):
        out = dw_conv_bn_relu(x, kernel_size, stride, dilation)
        out = conv_bn_relu(out, out_channel, 1, 1, 1)
        return out


class Model(object):
    def __init__(self, num_classes, name="MobileNetV1", data_format="channels_last", reuse=None):
        super(Model, self).__init__()
        self._name = name
        self._num_classes = num_classes
        self._data_format = data_format
        self._reuse = reuse
        self._decay = 0.997
        self._epsilon = 0.001
        self._fused = None
        self._use_bias = False
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        self._weight_init = tf.variance_scaling_initializer()
        self._bias_init = tf.zeros_initializer()
        self._kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)

    def forward(self, inputs, is_train=False):
        end_points = OrderedDict()
        batch_norm_params = {
            'is_training': is_train,
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'fused': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }
        with slim.arg_scope([slim.conv2d], weights_initializer=self._weight_init,
                            biases_initializer=self._bias_init):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope(self._name, reuse=self._reuse):
                    end_points['input'] = inputs
                    out = conv_bn_relu(inputs, 32, 3, 2)
                    end_points['init_unit'] = out
                    out = mobilenet1_unit(out, 64, 3, 1, name='mobilenet1_unit1')
                    end_points['mobilenet1_unit1'] = out
                    out = mobilenet1_unit(out, 128, 3, 2, name='mobilenet1_unit2')
                    end_points['mobilenet1_unit2'] = out
                    out = mobilenet1_unit(out, 128, 3, 1, name='mobilenet1_unit3')
                    end_points['mobilenet1_unit3'] = out
                    out = mobilenet1_unit(out, 256, 3, 2, name='mobilenet1_unit4')
                    end_points['mobilenet1_unit4'] = out
                    out = mobilenet1_unit(out, 256, 3, 1, name='mobilenet1_unit5')
                    end_points['mobilenet1_unit5'] = out
                    out = mobilenet1_unit(out, 512, 3, 2, name='mobilenet1_unit6')
                    end_points['mobilenet1_unit6'] = out
                    out = mobilenet1_unit(out, 512, 3, 1, name='mobilenet1_unit7')
                    end_points['mobilenet1_unit7'] = out
                    out = mobilenet1_unit(out, 512, 3, 1, name='mobilenet1_unit8')
                    end_points['mobilenet1_unit8'] = out
                    out = mobilenet1_unit(out, 512, 3, 1, name='mobilenet1_unit9')
                    end_points['mobilenet1_unit9'] = out
                    out = mobilenet1_unit(out, 512, 3, 1, name='mobilenet1_unit10')
                    end_points['mobilenet1_unit10'] = out
                    out = mobilenet1_unit(out, 512, 3, 1, name='mobilenet1_unit11')
                    end_points['mobilenet1_unit11'] = out
                    out = mobilenet1_unit(out, 1024, 3, 2, name='mobilenet1_unit12')
                    end_points['mobilenet1_unit12'] = out
                    out = mobilenet1_unit(out, 1024, 3, 1, name='mobilenet1_unit13')
                    end_points['mobilenet1_unit13'] = out
                    out = tf.reduce_mean(out, [1, 2], name='pool5', keepdims=True)
                    out = conv(out, self._num_classes, 1, 1)
                    out = tf.squeeze(out, [1, 2])
                    if not is_train:
                        out = slim.softmax(out, scope='predictions')
            return out


if __name__ == "__main__":
    input_data = tf.truncated_normal(shape=(1, 224, 224, 3), dtype=tf.float32)
    with tf.device('/cpu:0'):
        model = Model(num_classes=10, name="MobileNetV1", data_format="channels_last", reuse=None)
        output_data = model.forward(input_data, is_train=False)
    stats_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))
        sess.run(output_data)
        start_t = time.time()
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        sess.run(output_data)
        end_t = time.time()
        print("averge time: {} ms".format((end_t-start_t)/10*1000))
