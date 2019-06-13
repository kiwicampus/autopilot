import data_read
import tensorflow as tf 
import model

# Parameters

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


# Graph initializer

# tf Graph input
x = tf.placeholder(tf.float32, [None, 66, 200, 3])
y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

# Model
logits = model.autopilot(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.losses.mean_squared_error(labels=y, predictions = logits)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
accuracy = tf.metrics.mean_squared_error(labels = y, predictions = logits)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        batch = 0
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: X_validation[:test_valid_size],
                y: y_validation[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))
            batch += 1
    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: X_test[:test_valid_size],
        y: y_test[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))