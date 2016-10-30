#coding=utf-8

import tensorflow as tf 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


data_sets = read_data_sets("./data", one_hot=True)

#data_sets.train.images  55000x784  data_sets.train.labels 55000x1  data_sets.test.images 10000x784 data_sets.test.labels 10000x1  image 28x28
#data_sets.validation.images  5000x784 data_sets.validation.labels 5000x1


x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

mod = tf.reduce_sum(tf.abs(y-y_))				#通过最小化  E(|y`-y|) 求解  自创  效果差不多，时间复杂度低于cross_entropy   O(MN) < O(MMNN)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mod) 

#评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		batch_xs, batch_ys = data_sets.train.next_batch(100)
		sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})


	ret = sess.run(accuracy, feed_dict={x:data_sets.test.images, y_:data_sets.test.labels})
	print ret
