from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/root/data/",one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LearningRate=0.01
TrainingEpochs=25
BatchSize=100
DisplayStep=1;

hidden1=256
hidden2=256
inputNum=784
outputNum=10

x=tf.placeholder("float",[None,inputNum])
y=tf.placeholder("float",[None,outputNum])

def MultilayerPerception(x,weights,bias):
    layer1=tf.add(tf.matmul(x,weights['w1']),bias['b1'])
    layer1=tf.nn.relu(layer1)

    layer2=tf.add(tf.matmul(layer1,weights['w2']),bias['b2'])
    layer2=tf.nn.relu(layer2)

    layerOut=tf.matmul(layer2,weights['out'])+bias['out']

    return layerOut

weights={
    'w1':tf.Variable(tf.random_normal([inputNum,hidden1])),
    'w2':tf.Variable(tf.random_normal([hidden1,hidden2])),
    'out':tf.Variable(tf.random_normal([hidden2,outputNum]))

}
bias={

    'b1':tf.Variable(tf.random_normal([hidden1])),
    'b2':tf.Variable(tf.random_normal([hidden2])),
    'out':tf.Variable(tf.random_normal([outputNum]))
}

pred=MultilayerPerception(x,weights,bias)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y,pred))

optimizer=tf.train.AdamOptimizer(learning_rate=LearningRate).minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    times=np.linspace(1,25,25)
    outaccu=np.zeros(25)
    outcost=np.zeros(25)
    for epoch in range(1,TrainingEpochs+1):
        avg_cost=0.0

        for i in range(20):
            batchX,batchY=mnist.train.next_batch(BatchSize)
            
            _,c=sess.run([optimizer,cost],feed_dict={x:batchX,y:batchY})

            avg_cost=avg_cost+(c/20)

        outcost[int(epoch)-1]=avg_cost
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        outaccu[int(epoch)-1]=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})


print("最终损失：",outcost[-1],"最终精度：",outaccu[-1])

plt.subplot(211)
plt.xlim(xmax=25,xmin=0)
plt.plot(times,outcost)


plt.subplot(212)
plt.xlim(xmax=25,xmin=0)
plt.plot(times,outaccu)

plt.show()