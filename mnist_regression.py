import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#######
# Mnist 문제, Logistic Regression 으로 풀기
# 약 90.6 %의 정확도
########

# read dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# parameter
learning_rate = 0.5

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# weight & bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# operation
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tensorboard var #
tf.summary.scalar('cost', cross_entropy)
tf.summary.scalar("Training Accuracy", accuracy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./board/mnist_regression', sess.graph)

    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        summary,_ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
        writer.add_summary(summary, i)

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
