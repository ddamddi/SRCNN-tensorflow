import tensorflow as tf

w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001)
b_initializer = tf.constant_initializer(0)

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=w_initializer,
                             strides=stride, use_bias=use_bias, padding=padding)
        return x

def relu(x):
    return tf.nn.relu(x)

