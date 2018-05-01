import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

###
#   - initialize 기법 없이
#   - 경사하강법 대신에 AdamOptimizer 기법 사용
#   => 94.2%의 정확도
####
# read dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# weight & bias
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

# construct model
L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2)) # Hidden layer with ReLU activation
hypothesis = tf.add(tf.matmul(L2, W3), B3)     # No need to use softmax here

# minimize cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_prediction
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tensorboard var #
tf.summary.scalar('cost', cost)
tf.summary.scalar("Training Accuracy", accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./board/mnist_deeplearning1', sess.graph)

    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_,c = sess.run([merged, train_step, cost], feed_dict={x: batch_xs, y: batch_ys})
            writer.add_summary(summary, total_batch*epoch + i)
            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))