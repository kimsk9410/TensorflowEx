### cost and W check

# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# W = tf.placeholder(tf.float32)
# # Our hypothesis for linear model X * W
# hypothesis = X * W
#
# # cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
# # Launch the graph in a session.
# sess = tf.Session()
# # Initializes global variables in the graph
# sess.run(tf.global_variables_initializer())
# # Variables for plotting cost function
# W_val = []
# cost_val = []
#
# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)
#
# # Show the cost function
# plt.plot(W_val, cost_val)
# plt.show()

### cost minimize

import tensorflow as tf
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_sum(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= Learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X) # differential of (W * X - Y)^2
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))