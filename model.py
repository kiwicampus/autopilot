import tensorflow as tf
from tensorflow.contrib.layers import flatten

'''
Implements a 2D convolutional layer

'''
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

'''
Implements the autopilot architecture

'''
def autopilot(x, weights, biases, dropout):
    # Layer 1 - 66*200*3 to 31*98*24
    conv1 = conv2d(x, weights['wc1'], biases['bc1'] , 2)

    # Layer 2 - 31*98*24 to 14*47*36
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'] , 2)
    
    # Layer 3 - 14*47*36 to 5*22*48
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'] , 2)
    
    # Layer 4 - 5*22*48 to 3*20*64
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    
    # Layer 5 - 3*20*64 to 1*18*64
    conv5= conv2d(conv4, weights['wc5'], biases['bc5'])

    
    # Flatten feature map
    flat1 = flatten(conv5)
    
    # Fully connected layer - 1*18*64 to 100
    fc1 = tf.add(tf.matmul(flat1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    # Fully connected layer - From 100 to 50
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, dropout)
    
    # Fully connected layer - From 50 to 10
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    
    # Fully connected layer - Output Layer - class prediction - 10 to 1
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out