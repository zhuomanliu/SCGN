import tensorflow as tf

def batch_norm(x, eps=1e-5, mom=0.9, name="batch_norm", train=True):
    return tf.contrib.layers.batch_norm(x, decay=mom, #updates_collections=None,
                                        epsilon=eps, is_training=train, scope=name)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def atrous_conv2d(input_, output_dim,
                  k_h=3, k_w=3, rate=1, stddev=0.02,
                  name="atrous_conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(input_, w, rate, padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def resblock(input_, k_h=3, k_w=3, d_h=1, d_w=1, name = "resblock"):
    conv1 = lrelu(conv2d(input_, input_.get_shape()[-1],
                                    k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                                    name=name + "_conv1"))
    conv2 = conv2d(conv1, input_.get_shape()[-1],
                            k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                            name=name + "_conv2")
    return tf.add(input_, conv2)

def resblock_relu(input_, k_h=3, k_w=3, d_h=1, d_w=1, name = "resblock"):
    conv1 = tf.nn.relu(conv2d(input_, input_.get_shape()[-1],
                                    k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                                    name=name + "_conv1"))
    conv2 = conv2d(conv1, input_.get_shape()[-1],
                            k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                            name=name + "_conv2")
    return tf.add(input_, conv2)