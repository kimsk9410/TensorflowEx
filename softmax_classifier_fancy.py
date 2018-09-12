import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0 : -1]
y_data = xy[:, [-1]]

print("x_data.shape : ", x_data.shape, " / y_data.shape : ", y_data.shape)

class_num = 7 # 데이터의 클래스가 0 ~ 6 의 값을 가짐

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

# Y -> Y_one_hot
# 0 ~ 6 의 값을 가지는 변수를 One-hot 값을 가지는 변수로 변환하는 과정
Y_one_hot = tf.one_hot(Y, class_num) # 여기서 Y_one_hot 의 shape에 주의. (2차원 데이터가 3차원으로 변환됨)
print("one_hot : ", Y_one_hot)

# (?, 1, 7) -> (?, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, class_num])
print("reshape !")
print("one_hot : ", Y_one_hot)

W = tf.Variable(tf.random_normal([16, class_num]), name='weight')
b = tf.Variable(tf.random_normal([class_num]), name='bias')

# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
# Cross entropy를 직접 쓰지 않고 텐서플로우의 함수 사용.
# logits 과 Y_one_hot 만 인자로 넘겨준다.
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 정확도를 구하기 위해 예측값 저장.
prediction = tf.argmax(hypothesis, 1)
# 예측한 class와 실제 class가 같은지 비교.
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))