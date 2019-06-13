import data_read
import model
    
# Graph initializer

# tf Graph input
x = tf.placeholder(tf.float32, [None, 66, 200, 3)
y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

# Model
logits = autopilot(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.losses.mean_squared_error(labels=y, predictions = logits)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
accuracy = tf.metrics.mean_squared_error(labels = y, predictions = logits)

# Initializing the variables
init = tf.global_variables_initializer()