import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Parameters
learning_rate = 0.00001
epochs = 100
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 100

# Network Parameters
dropout = 0.75  # Dropout, probability to keep units

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 24, 36])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 36, 48])),
    
    'wc4': tf.Variable(tf.random_normal([3, 3, 48, 64])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    
    'wd1': tf.Variable(tf.random_normal([64*18, 100])),
    'wd2': tf.Variable(tf.random_normal([100, 50])),
    'wd3': tf.Variable(tf.random_normal([50, 10])),
    'out': tf.Variable(tf.random_normal([10, 1]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([24])),
    'bc2': tf.Variable(tf.random_normal([36])),
    'bc3': tf.Variable(tf.random_normal([48])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([64])),
      
    'bd1': tf.Variable(tf.random_normal([100])),
    'bd2': tf.Variable(tf.random_normal([50])),
    'bd3': tf.Variable(tf.random_normal([10])),
    'out': tf.Variable(tf.random_normal([1]))}

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