import tensorflow as tf 
from functools import partial
import sys
import numpy as np
import config

class MobileNet():
    def __init__(self, num_class, is_training, num_bits=None, width_multiplier=1, quant_mode='tensorflow', conv2d_regularizer=None):
        self.add_fake_quant = quant_mode=='custom' and num_bits is not None
        self.num_class = num_class
        self.num_bits = num_bits
        self.conv2d_regularizer=conv2d_regularizer
        self.relu6 = partial(self._relu6, num_bits)
        self.conv2d = partial(self._conv2d, num_bits)
        self.depthwise_conv2d = partial(self._depthwise_conv2d, num_bits)
        self.is_training = is_training
        self.width_multiplier = width_multiplier
        tf.logging.info("Creating graph. is_training=%s, width multiplier=%.2f add_fake_quant=%s"%(self.is_training, self.width_multiplier, self.add_fake_quant))
    def _relu6(self, num_bits, x):
        with tf.variable_scope("act"):
            x = tf.nn.relu6(x)
            if self.add_fake_quant:
                x = tf.fake_quant_with_min_max_vars(x, 0.0, 6.0, num_bits)
        return x

    def _conv2d(self, 
            num_bits,
            x,             
            n_output_plane,
            kernel_size,
            strides=1,
            bias=False,
            padding='SAME',
            name='conv2d'):
        with tf.variable_scope(name):
            n_input_plane = x.get_shape().as_list()[3]
            w_dim = [kernel_size, kernel_size, n_input_plane, n_output_plane]
            w = tf.get_variable("weight", w_dim, 
                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                regularizer=self.conv2d_regularizer)
            if self.add_fake_quant:
                w_min = tf.reduce_min(w)
                w_max = tf.reduce_max(w)
                w = tf.fake_quant_with_min_max_vars(w, w_min, w_max, num_bits)

            output = tf.nn.conv2d(x, w, [1, strides, strides,1], padding)

            if bias:
                b = tf.get_variable('bias', [n_output_plane])
                output =  tf.nn.bias_add(output, b)
        #tf.logging.info('conv2d output tensor: %s'%x.get_shape())

        return output

    def _depthwise_conv2d(self, 
            num_bits,
            x,             
            n_output_plane,
            kernel_size,
            strides=1,
            bias=False,
            padding='SAME',
            name='depthwise_conv2d'):

        with tf.variable_scope(name):
            n_input_plane = x.get_shape().as_list()[3]
            w_dim = [kernel_size, kernel_size, n_output_plane, 1]
            w = tf.get_variable("weight", w_dim, 
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            if self.add_fake_quant:
                w_min = tf.reduce_min(w)
                w_max = tf.reduce_max(w)
                w = tf.fake_quant_with_min_max_vars(w, w_min, w_max, num_bits)

            output = tf.nn.depthwise_conv2d(x, w, [1, strides, strides,1], padding)

            if bias:
                b = tf.get_variable('bias', [n_output_plane])
                output =  tf.nn.bias_add(output, b)
        #tf.logging.info('depthwise output tensor: %s'%x.get_shape())
        return output

    def separable_conv2d(self, 
            x,             
            kernel_size,
            filters,
            strides=1,
            bias=False,
            padding='SAME',
            name='separable_conv2d'):
        with tf.variable_scope(name):
            filter_depth = self.width_multiplier*np.array(filters)
            x = self.depthwise_conv2d(x, 
                    int(filter_depth[0]), 
                    kernel_size, 
                    strides, 
                    bias, 
                    padding)

            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = self.relu6(x)

            x = self.conv2d(x, 
                    int(filter_depth[1]), 
                    kernel_size=1, 
                    strides=1, 
                    bias=bias, 
                    padding=padding)
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = self.relu6(x)
        return x

    def forward_pass(self, x):

        filter_depth = np.int(self.width_multiplier*32)
        x = self.conv2d(x, filter_depth, 3, 2)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = self.relu6(x)

        x = self.separable_conv2d(x, 3, [32, 64], 1, name='separable_1')
        x = self.separable_conv2d(x, 3, [64, 128], 2, name='separable_2')
        x = self.separable_conv2d(x, 3, [128, 128], 1, name='separable_3')
        x = self.separable_conv2d(x, 3, [128, 256], 2, name='separable_4')
        x = self.separable_conv2d(x, 3, [256, 256], 1, name='separable_5')
        x = self.separable_conv2d(x, 3, [256, 512], 2, name='separable_6')
        x = self.separable_conv2d(x, 3, [512, 512], 1, name='separable_7')
        x = self.separable_conv2d(x, 3, [512, 512], 1, name='separable_8')
        x = self.separable_conv2d(x, 3, [512, 512], 1, name='separable_9')
        x = self.separable_conv2d(x, 3, [512, 512], 1, name='separable_10')
        x = self.separable_conv2d(x, 3, [512, 512], 1, name='separable_11')
        x = self.separable_conv2d(x, 3, [512, 1024], 2, name='separable_12')
        x = self.separable_conv2d(x, 3, [1024, 1024], 1, name='separable_13')

        pool_size = x.get_shape().as_list()[1]
        x = tf.layers.average_pooling2d(x, pool_size=pool_size, strides=1)
        x = tf.layers.dropout(x, training=self.is_training)
        x = self.conv2d(x, self.num_class, 1,1, name='fc')
        x = tf.squeeze(x, [1, 2], 'spatial_squeeze')

        x_shape = x.get_shape().as_list()
        tf.logging.info("output tensor: %s", x.get_shape())
        return x
