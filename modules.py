'''
Author: Yue Li

This file contains low-level custom layers.
'''

import tensorflow as tf
import numpy as np
from params import Params as param

class ResUnit(tf.keras.layers.Layer):
    '''
    Defines a residual unit
    input -> [batchnorm->relu->conv] X 2 -> reslink -> output
    '''
    def __init__(self, filters, kernel_size, strides, batchnorm=True, **kwargs):
        super(ResUnit, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.batchnorm_TF = batchnorm

    def call(self, inputs, **kwargs):
        output = inputs
        if self.batchnorm_TF:
            output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.conv1(output)
        if self.batchnorm_TF:
            output = self.batchnorm2(output)
        output = self.relu2(output)
        output = self.conv2(output)
        output += inputs
        return output


class ResNet(tf.keras.layers.Layer):
    '''
    Defines the loop of ResUnit between the first and the last layer of the ResNet architecture
    '''
    def __init__(self, filters, kernel_size, num_units, strides=1, batchnorm=True, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_units = num_units
        self.strides = strides
        self.batchnorm = batchnorm
        self.res_units = [ResUnit(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, batchnorm=self.batchnorm) for _ in range(self.num_units)]

    def call(self, inputs, **kwargs):
        output = inputs
        for res_unit in self.res_units:
            output = res_unit(output)
        return output


    
class ResInput(tf.keras.layers.Layer):
    '''
    Defines the first (input) layer of the ResNet architecture
    '''
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResInput, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        return output

    
class ResOutput(tf.keras.layers.Layer):
    '''
    Defines the last (output) layer of the ResNet architecture
    '''
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResOutput, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        return output
        
        
class Fusion(tf.keras.layers.Layer):
    
    def __init__(self):
        super(Fusion, self).__init__()
        
    def build(self, input_shape):
        shape = input_shape[1:]
        self.Wc = self.add_weight(shape=shape, initializer='glorot_uniform', trainable=True, name="closeness_matrix")
        self.Wp = self.add_weight(shape=shape, initializer='glorot_uniform', trainable=True, name="period_matrix")
        self.Wt = self.add_weight(shape=shape, initializer='glorot_uniform', trainable=True, name="trend_matrix")

    def call(self, c, p, t):
        closeness = tf.multiply(c, self.Wc)
        period = tf.multiply(p, self.Wp)
        trend = tf.multiply(t, self.Wt)
        outputs = tf.add(tf.add(closeness, period), trend)
        return outputs
