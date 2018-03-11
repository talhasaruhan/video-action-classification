import tensorflow as tf

import numpy as np
from functools import reduce


class CnnM:
    """
    CNN-M-2048 implementation
    """

    def __init__(self, L, dropout=0.6):
        self.var_dict = {}
        self.L = L
        self.data_dict = None
        self.trainable = True
        self.dropout = dropout
        self.layers = []

    def build(self, bgr, useDropout=None):
        """
        load variable from npy to build the VGG

        :param bgr: bgr image [batch, height, width, channel_depth=L] values scaled [0, 255]
        :param useDropout: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # optical flow was converted from [-bound, bound] to [0, 255]
        # subtract the "gray" mean
        bgr = bgr - 255/2
        self.conv1 = self.conv_layer(bgr, self.L, 96, kernel_size=7, stride=2, name="conv1")
        self.pool1 = self.max_pool(self.conv1, name='pool1')

        self.conv2 = self.conv_layer(self.pool1, 96, 256, kernel_size=5, stride=2, name="conv2")
        self.pool2 = self.max_pool(self.conv2, name='pool2')

        self.conv3 = self.conv_layer(self.pool2, 256, 512, kernel_size=3, stride=1, name="conv3")
        self.conv4 = self.conv_layer(self.conv3, 512, 512, kernel_size=3, stride=1, name="conv4")
        self.conv5 = self.conv_layer(self.conv4, 512, 512, kernel_size=3, stride=1, name="conv5")
        self.pool3 = self.max_pool(self.conv5, name='pool3')

        self.fc1 = self.fc_layer(self.pool3, 25088, 4096, "fc1")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu1 = tf.nn.relu(self.fc1)
        if useDropout is not None:
            self.relu1 = tf.cond(useDropout, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        elif self.trainable:
            self.relu1 = tf.nn.dropout(self.relu1, self.dropout)

        self.fc2 = self.fc_layer(self.relu1, 4096, 2048, "fc2")
        self.relu2 = tf.nn.relu(self.fc2)
        if useDropout is not None:
            self.relu2 = tf.cond(useDropout, lambda: tf.nn.dropout(self.relu2, self.dropout), lambda: self.relu2)
        elif self.trainable:
            self.relu2 = tf.nn.dropout(self.relu2, self.dropout)

    def avg_pool(self, bottom, kernel_size=2, stride=2, name=None):
        return tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def max_pool(self, bottom, kernel_size=2, stride=2, name=None):
        return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, kernel_size=3, stride=2, name=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernel_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            self.layers.append(conv)
            self.layers.append(relu)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            self.layers.append(fc)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        weights = self.get_var(name + "_weights", [filter_size, filter_size, in_channels, out_channels])
        biases = self.get_var(name + "_biases", [out_channels])

        return weights, biases

    def get_fc_var(self, in_size, out_size, name):
        weights = self.get_var(name + "_weights", [in_size, out_size])
        biases = self.get_var(name + "_biases", [out_size])

        return weights, biases

    def get_var(self, name, shape):
        if self.data_dict is not None and name in self.data_dict:
            placeholder = tf.placeholder(tf.float32, shape)
            var = tf.Variable(placeholder, name=name)
            self.var_dict[placeholder] = self.data_dict[name]
        else:
            var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(0.0, 0.001))

        return var

    def print(self):
        for layer in self.layers:
            print(layer)

    def save_npy(self, sess, npy_path="./cnnm-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count