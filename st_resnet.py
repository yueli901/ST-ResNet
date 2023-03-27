'''
Author: Yue Li

This file defines the Tensorflow computation graph for the ST-ResNet (Deep Spatio-temporal Residual Networks) architecture.
'''

import tensorflow as tf
from params import Params as param
import modules as my

class ST_ResNet(tf.keras.Model):
    def __init__(self):
        super(ST_ResNet, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=param.lr, beta_1=param.beta1, beta_2=param.beta2, epsilon=param.epsilon)
        
        F, U  = param.num_of_filters, param.num_of_residual_units
        
        # ResNet architecture for the three modules
        self.closeness_input = my.ResInput(filters=F, kernel_size=(3, 3))
        self.closeness_resnet = my.ResNet(filters=F, kernel_size=(3, 3), num_units=U, strides=1, batchnorm=True)
        self.closeness_output = my.ResOutput(filters=param.num_of_output, kernel_size=(3, 3))
        
        self.period_input = my.ResInput(filters=F, kernel_size=(3, 3))
        self.period_resnet = my.ResNet(filters=F, kernel_size=(3, 3), num_units=U, strides=1, batchnorm=True)
        self.period_output = my.ResOutput(filters=param.num_of_output, kernel_size=(3, 3))
        
        self.trend_input = my.ResInput(filters=F, kernel_size=(3, 3))
        self.trend_resnet = my.ResNet(filters=F, kernel_size=(3, 3), num_units=U, strides=1, batchnorm=True)
        self.trend_output = my.ResOutput(filters=param.num_of_output, kernel_size=(3, 3))
        
        self.fusion = my.Fusion()
    
    def call(self, c_inp, p_inp, t_inp):
        closeness = self.closeness_input(c_inp)
        closeness = self.closeness_resnet(closeness)
        closeness = self.closeness_output(closeness)
        
        period = self.period_input(p_inp)
        period = self.period_resnet(period)
        period = self.period_output(period)
        
        trend = self.trend_input(t_inp)
        trend = self.trend_resnet(trend)
        trend = self.trend_output(trend)
        
        output = self.fusion(closeness, period, trend)
        return output
    
    def train_step(self, data):
        x, y = data
        x_closeness, x_period, x_trend = x
        with tf.GradientTape() as tape:
            y_pred = self.call(x_closeness, x_period, x_trend)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss_train": loss}

    def test_step(self, data):
        x, y = data
        x_closeness, x_period, x_trend = x
        y_pred = self.call(x_closeness, x_period, x_trend)
        loss = tf.reduce_mean(tf.square(y_pred - y))
        return {"loss_test": loss}