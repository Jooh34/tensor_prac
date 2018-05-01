import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
###
#   - initialize : 표준편차 0.1 랜덤 var , bias 사용 x
#   - convolutional neural network + relu + dropout
#   - RMSPropOptimizer
#   - accuracy : 99.19%
####

def init_weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

start_time = time.time()

# read dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

### model definition ###

l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, p_keep_conv)

l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, p_keep_conv)

l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, p_keep_hidden)

hypothesis = tf.matmul(l4, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########################

# tensorboard var #
tf.summary.scalar('cost', cost)
tf.summary.scalar("Training Accuracy", accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./board/mnist_CNN', sess.graph)
    sess.run(init)

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        x_train = batch[0].reshape(-1,28,28,1)
        y_train = batch[1]

        summary, _ = sess.run([merged, train_op], feed_dict={X: x_train, Y: y_train, p_keep_conv: 0.8, p_keep_hidden: 0.5})
        writer.add_summary(summary, i)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                X: x_train, Y: y_train, p_keep_conv: 0.8, p_keep_hidden: 0.5})
            print("step %d, training accuracy %g" % (i, train_accuracy))

    x_test = mnist.test.images.reshape(-1,28,28,1)
    y_test = mnist.test.labels
    print("test accuracy %g" % accuracy.eval(feed_dict={
        X: x_test, Y: y_test, p_keep_conv: 1.0, p_keep_hidden: 1.0}))

print("running time : %s seconds" %(time.time() - start_time))