
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import matplotlib.pyplot as plt


def build_data(n):
    xs = []
    ys = []
    data=[]
    cc=np.linspace(1,607,607)
    data=np.sin(0.06*cc)+np.random.uniform(-0.1,0.1,607)
    for i in range(0, 600):
        x = [[data[i + j]] for j in range(0, n)]
        y = [data[i + n]]

        xs.append(x)
        ys.append(y)

    train_x = np.array(xs[0: 420])
    train_y = np.array(ys[0: 420])
    test_x = np.array(xs[420:])
    test_y = np.array(ys[420:])
    return (data,train_x, train_y, test_x, test_y)


length = 7
time_step_size = length
vector_size = 1
batch_size = 10
test_size = 10

# build data
(data,train_x, train_y, test_x, test_y) = build_data(length)
plt.subplot(221)
plt.plot(data)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

X = tf.placeholder("float", [None, length, vector_size])
Y = tf.placeholder("float", [None, 1])

# get lstm_size and output predicted value
W = tf.Variable(tf.random_normal([7, 1], stddev=0.01))
B = tf.Variable(tf.random_normal([1], stddev=0.01))


def seq_predict_model(X, w, b, time_step_size, vector_size):
    # input X shape: [batch_size, time_step_size, vector_size]
    # transpose X to [time_step_size, batch_size, vector_size]
    X = tf.transpose(X, [1, 0, 2])
    # reshape X to [time_step_size * batch_size, vector_size]
    X = tf.reshape(X, [-1, vector_size])
    # split X, array[time_step_size], shape: [batch_size, vector_size]
    X = tf.split(X, time_step_size, 0)

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=7)
    initial_state = tf.zeros([batch_size, cell.state_size])
    outputs, _states = tf.contrib.rnn.static_rnn(cell, X, initial_state=initial_state)

    # Linear activation
    return tf.matmul(outputs[-1], w) + b, cell.state_size


pred_y, _ = seq_predict_model(X, W, B, time_step_size, vector_size)
loss = tf.square(tf.subtract(Y, pred_y))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # train
    loss1=[]
    for i in range(50):
        # train
        for end in range(batch_size, len(train_x), batch_size):
            begin = end - batch_size
            x_value = train_x[begin: end]
            y_value = train_y[begin: end]
            sess.run(train_op, feed_dict={X: x_value, Y: y_value})

        # randomly select validation set from test set
        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        x_value = test_x[test_indices]
        y_value = test_y[test_indices]

        # eval in validation set
        val_loss=np.mean(sess.run(loss,feed_dict={X: x_value, Y: y_value}))
        loss1.append(val_loss)
    print('最终误差：',loss1[-1])
    plt.subplot(223)
    plt.plot(loss1)
    ccpred=[]
    for b in range(0, len(test_x), test_size):
        x_value = test_x[b: b + test_size]
        y_value = test_y[b: b + test_size]
        pred = sess.run(pred_y, feed_dict={X: x_value})
        for i in range(len(pred)):
            ccpred.append(pred[i])

    plt.subplot(224)
    plt.plot(ccpred)
    plt.plot(test_y)
    plt.show()