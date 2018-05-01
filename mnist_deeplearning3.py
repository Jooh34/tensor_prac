import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

###
#   - initialize 기법 : xavier
#   - 경사하강법 대신에 AdamOptimizer 기법 사용
#   - hidden layer 3개와 dropout 기법 추가
#   => 정확도 98.32%
####

# read dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# parameter
learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder(tf.float32)

# weight & bias
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape=[256,  10], initializer=tf.contrib.layers.xavier_initializer())

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([ 10]))

# Construct model
_L1 = tf.nn.relu(tf.add(tf.matmul(x,W1),B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),B2)) # Hidden layer with ReLU activation
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3),B3)) # Hidden layer with ReLU activation
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4),B4)) # Hidden layer with ReLU activation
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), B5)

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
    writer = tf.summary.FileWriter('./board/mnist_deeplearning3', sess.graph)

    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_,c = sess.run([merged, train_step, cost], feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7})
            writer.add_summary(summary, total_batch*epoch + i)
            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1.0}))