import numpy as np
import tensorflow as tf

def batch_generator(X, y, batch_size):
    """
    Batch generator 
    The function repeatedly yields batches of the specified size, randomly shuffling the data once an epoch is completed.
    @input: all (X,y) pairs for train data
        X: [[(train/test size,32,32,6)]*3,(train/test size,external_dim)]
        y: (train/test size,32,32,2)
        batch_size: default to 32
    @Output: a batch of (X,y) pairs with the same shape as input
    """
    size = y.shape[0]
    indices = np.arange(size)
    i = 0
    while True:
        if i + batch_size <= size:
            xc_batch = X[0][i:i+batch_size]
            xp_batch = X[1][i:i+batch_size]
            xt_batch = X[2][i:i+batch_size]
            ext_batch = X[3][i:i+batch_size]
            y_batch = y[i:i+batch_size]
            yield [xc_batch, xp_batch, xt_batch, ext_batch], y_batch
            i += batch_size
        else:
            i = 0
            np.random.shuffle(indices)
            continue
