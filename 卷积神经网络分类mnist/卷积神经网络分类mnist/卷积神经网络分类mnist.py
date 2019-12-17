import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
LearningRate=0.0001

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def ConvolutionLayer(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def MaxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

x_image=tf.reshape(x,[-1,28,28,1])
w_conv1=weight([5,5,1,32])
b_conv1=bias([32])

h_conv1=tf.nn.relu(tf.add(ConvolutionLayer(x_image,w_conv1),b_conv1))
h_pool1=MaxPool(h_conv1)

w_conv2=weight([5,5,32,64])
b_conv2=bias([64])

h_conv2=tf.nn.relu(tf.add(ConvolutionLayer(h_pool1,w_conv2),b_conv2))
h_pool2=MaxPool(h_conv2)

w_fullConnect1=weight([7*7*64,1024])
b_fullConnect1=bias([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

h_fullConnect1=tf.nn.relu(tf.add(tf.matmul(h_pool2_flat,w_fullConnect1),b_fullConnect1))

keep_prob=tf.placeholder(tf.float32)

h_fullConnect1_drop=tf.nn.dropout(h_fullConnect1,keep_prob)

w_fullConnect2=weight([1024,10])
b_fullConnect2=bias([10])

pred=tf.nn.softmax(tf.add(tf.matmul(h_fullConnect1_drop,w_fullConnect2),b_fullConnect2))

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred))

optimizer=tf.train.AdamOptimizer(LearningRate).minimize(cost)

correct_prediction=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    times=np.linspace(1,25,25)
    outaccu=np.zeros(25)
    outcost=np.zeros(25)
    for epoch in range(1,26):
        avg_cost=0.0
        for batch in range(10):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.75})
            avg_cost=avg_cost+(c/10)

        outcost[int(epoch)-1]=avg_cost
        outaccu[int(epoch)-1]=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print('epoch:',epoch)
print("最终损失：",outcost[-1],"最终精度：",outaccu[-1])

plt.subplot(211)
plt.xlim(xmax=25,xmin=0)
plt.plot(times,outcost)


plt.subplot(212)
plt.xlim(xmax=25,xmin=0)
plt.plot(times,outaccu)

plt.show()
