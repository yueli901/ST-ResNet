'''
Author: Yue Li

This file defines a batch_generator to draw a batch of data.
'''
import numpy as np
import tensorflow as tf

def batch_generator(X, y, batch_size):
    """
    Batch generator 
    """
    size = X.shape[0]
    indices = np.arange(size)
    i = 0
    while True:
        if i + batch_size <= size:
            x_batch = tf.gather(X, tf.constant(indices[i:i + batch_size]), axis=0)
            y_batch = tf.gather(y, tf.constant(indices[i:i + batch_size]), axis=0)
            yield x_batch, y_batch
            i += batch_size
        else:
            i = 0
            np.random.shuffle(indices)
            continue
