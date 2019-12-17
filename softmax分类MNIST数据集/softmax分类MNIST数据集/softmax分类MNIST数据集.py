import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


#模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#构建模型
predict_y = tf.nn.softmax(tf.matmul(x, W) + b)
learning_rate=0.01

#损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_y)))

#梯度下降法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化所有变量
init=tf.global_variables_initializer()

#tf.argmax()返回最大值所在的列，结果存放在一个bool型列表中
correct_prediction=tf.equal(tf.argmax(predict_y,1),tf.argmax(y,1))

#准确率
accuary=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(init)
    times=np.linspace(0,999,1000)
    outaccu=np.zeros(1000)
    outcost=np.zeros(1000)


    for i in times:
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})

        #每次分类损失平均值
        outcost[int(i)]=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys})

        #每次分类精度
        outaccu[int(i)]=sess.run(accuary,feed_dict={x:mnist.test.images,y:mnist.test.labels})

    print("最终分类精度：")
    print(outaccu[-1])
    print("最终损失：")
    print(outcost[-1])

plt.subplot(211)
plt.xlim(xmax=1000,xmin=0)
plt.ylim(ymax=1,ymin=0)
plt.plot(times,outaccu)


plt.subplot(212)
plt.xlim(xmax=1000,xmin=0)
plt.plot(times,outcost)

plt.show()