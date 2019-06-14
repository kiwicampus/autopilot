import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call

sess = tf.InteractiveSession()
def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)

# Graph initializer
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

# tf Graph input
x = tf.placeholder(tf.float32, [None, 66, 200, 3], name="myInput")
y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

# Model
logits = model.autopilot(x, weights, biases, keep_prob)

addNameToTensor(logits, "myOutput")

saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = logits.eval(feed_dict={x: [image], keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()