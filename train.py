import data_read
import tensorflow as tf 
import model

# Data
num_examples = data_read.num_train_images

# Parameters
learning_rate = 0.00001
epochs = 30
batch_size = 100

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 100

# Network Parameters - In case dropout layer are added to the network architecture
dropout = 0.75  # Dropout, probability to keep units

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 24], stddev=0.1)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 24, 36], stddev=0.1)),
    'wc3': tf.Variable(tf.truncated_normal([5, 5, 36, 48], stddev=0.1)),
    
    'wc4': tf.Variable(tf.truncated_normal([3, 3, 48, 64], stddev=0.1)),
    'wc5': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
    
    'wd1': tf.Variable(tf.truncated_normal([64*18, 100], stddev=0.1)),
    'wd2': tf.Variable(tf.truncated_normal([100, 50], stddev=0.1)),
    'wd3': tf.Variable(tf.truncated_normal([50, 10], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))}


biases = {
        'bc1': tf.Variable(tf.random_normal([24])),
        'bc2': tf.Variable(tf.random_normal([36])),
        'bc3': tf.Variable(tf.random_normal([48])),
        'bc4': tf.Variable(tf.random_normal([64])),
        'bc5': tf.Variable(tf.random_normal([64])),
        
        'bd1': tf.Variable(tf.random_normal([100])),
        'bd2': tf.Variable(tf.random_normal([50])),
        'bd3': tf.Variable(tf.random_normal([10])),
        'out': tf.Variable(tf.random_normal([1]))
        }


# Graph initializer

# tf Graph input
x = tf.placeholder(tf.float32, [None, 66, 200, 3])
y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

# Model
logits = model.autopilot(x, weights, biases, keep_prob)

L2NormConst = 0.001

train_vars = tf.trainable_variables()

# Define loss and optimizer
#cost = tf.losses.mean_squared_error(labels=y, predictions = logits)
cost = tf.reduce_mean(tf.square(tf.subtract(y,logits)))  + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
# Accuracy
#accuracy = tf.metrics.mean_squared_error(labels = y, predictions = logits)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        batch = 0
        #X_train, y_train = shuffle(X_train, y_train)
        for i in range(int(num_examples/batch_size)):
            batch_x, batch_y = data_read.LoadTrainBatch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            X_validation, y_validation = data_read.LoadValBatch(test_valid_size)
            valid_acc = sess.run(cost,feed_dict={x:X_validation, y: y_validation, keep_prob: 1.0}) 
            '''
            sess.run(accuracy, feed_dict={
            x: X_validation[:test_valid_size],
            y: y_validation[:test_valid_size],
            keep_prob: 1.})
            '''
            print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))
            batch += 1
    '''
    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: X_test[:test_valid_size],
        y: y_test[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
    '''