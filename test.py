#coding=utf-8
import tensorflow as tf

#1. 做一个简单的矩阵相乘
'''
matrix1 = tf.constant([[3, 1]])   #1x2
matrix2 = tf.constant([[3], [3]])  #2x1

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
	ret = sess.run(product)
	print ret

'''

#2. 做一个变量叠加，测试变量功能

'''
state = tf.Variable(0, "counter")

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init_op)

	# print sess.run(state)

	for _ in range(3):
		print sess.run(update)
'''

#3. 测试fetch，run 中传入tensor获取多个返回

input1 = tf.placeholder(tf.int32)
input2 = tf.constant(3)
input3 = tf.constant(5)

add = tf.add(input1, input2)
mul = tf.mul(input3, add)

with tf.Session() as sess:
	ret = sess.run([mul, add], feed_dict={input1: 3})
	print ret