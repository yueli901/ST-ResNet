import tensorflow as tf

class ResUnit(tf.keras.layers.Layer):
    '''
    Defines a residual unit.
    input -> [(batchnorm) -> relu -> conv] *2 -> reslink(+=input) -> output
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
    Defines the loop of ResUnit.
    input -> [ResUnit] *num_units -> output
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
    Defines the first (input) layer of the ResNet architecture.
    input -> conv -> output
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
    Defines the last (output) layer of the ResNet architecture.
    input -> conv -> output
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
    '''
    Defines the fusion process in the ResNet architecture.
    input(closenss, period, trend) -> Wc*closeness + Wp*period + Wt*trend -> output
    '''
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
    
    
class AddExternal(tf.keras.layers.Layer):
    '''
    Add the external variables to the output.
    input(shape = (None,28) by default) -> dense(None,10) -> dense(None,32*32*2) -> reshape(None,32,32,2) -> output
    '''
    def __init__(self, embedding_dim=10, target_shape=(32,32,2), **kwargs):
        super(AddExternal, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.target_shape = tuple(target_shape)
        self.dense1 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(tf.reduce_prod(self.target_shape), activation='relu')
        
    def call(self, inputs, **kwargs):
        output = self.dense1(inputs)
        output = self.dense2(output)
        output = tf.reshape(output, (-1,)+self.target_shape)
        return output
    
    

class ST_ResNet(tf.keras.Model):
    '''
    Build the graph using above modules.
    closeness -> ResInput -> ResNet -> ResOutput -|
    period    -> ResInput -> ResNet -> ResOutput -|=> Fusion -|=> tanh -> output
    trend     -> ResInput -> ResNet -> ResOutput -|           |
    external  -> AddExternal ---------------------------------|
    '''
    def __init__(self,lr=0.0002,nb_filters=64,nb_flow=2,nb_residual_unit=4,batchnorm=False):
        super(ST_ResNet, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.8, beta_2=0.999, epsilon=1e-7)

        # ResNet architecture for the three modules
        self.closeness_input = ResInput(filters=nb_filters, kernel_size=(3, 3))
        self.closeness_resnet = ResNet(filters=nb_filters, kernel_size=(3, 3), num_units=nb_residual_unit, strides=1, batchnorm=batchnorm)
        self.closeness_output = ResOutput(filters=nb_flow, kernel_size=(3, 3))
        
        self.period_input = ResInput(filters=nb_filters, kernel_size=(3, 3))
        self.period_resnet = ResNet(filters=nb_filters, kernel_size=(3, 3), num_units=nb_residual_unit, strides=1, batchnorm=batchnorm)
        self.period_output = ResOutput(filters=nb_flow, kernel_size=(3, 3))
        
        self.trend_input = ResInput(filters=nb_filters, kernel_size=(3, 3))
        self.trend_resnet = ResNet(filters=nb_filters, kernel_size=(3, 3), num_units=nb_residual_unit, strides=1, batchnorm=batchnorm)
        self.trend_output = ResOutput(filters=nb_flow, kernel_size=(3, 3))
        
        self.fusion = Fusion()
        self.add_external = AddExternal(embedding_dim=10, target_shape=(32,32,2)) 
        
        # Define the training and testing loss loggers
        self.train_loss_logger = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss_logger = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    
    def call(self, c_inp, p_inp, t_inp, ext=None):
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
        
        if ext is not None:
            ext_output = self.add_external(ext)
            output += ext_output
        
        output = tf.keras.activations.tanh(output)
        return output
    
    @tf.function
    def train_step(self, data):
        x, y = data
        x_closeness, x_period, x_trend, ext = x
        with tf.GradientTape() as tape:
            y_pred = self.call(x_closeness, x_period, x_trend, ext)
            y = tf.cast(y, tf.float32)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss_logger.update_state(loss)

    @tf.function
    def test_step(self, data, mmn):
        x, y = data
        x_closeness, x_period, x_trend, ext = x
        y_pred = self.call(x_closeness, x_period, x_trend, ext)
        y = tf.cast(y, tf.float32)
        y_pred = mmn.inverse_transform(y_pred)
        y = mmn.inverse_transform(y)
        loss = tf.reduce_mean(tf.square(y_pred - y))
        self.test_loss_logger.update_state(loss)
    