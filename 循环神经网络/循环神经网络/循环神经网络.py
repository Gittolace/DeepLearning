import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn


class Data:
    def __init__(self):
        self.batch_size=7
        self.num_batch=413
        self.timestep=7
        self.x1=np.linspace(1,600,600)
        self.y1=np.sin(0.06*self.x1)+np.random.uniform(-0.1,0.1,600)

    def generate_epochs(self):
        data_x=np.zeros([self.num_batch,self.batch_size])
        data_y=np.zeros([self.num_batch,self.batch_size])
        for i in range(413):
            data_x[i]=self.x1[i:i+7]
            data_y[i]=self.y1[i+7:i+14]

        epoch_size=self.batch_size//self.timestep

        for i in range(epoch_size):
            x=data_x[:,self.timestep*i:self.timestep*(i+1)]
            y=data_y[:,self.timestep*i:self.timestep*(i+1)]
            yield(x,y)

class Model:
    def __init__(self):
        self.batch_size=7
        self.num_batch=413
        self.timestep=7
        self.state_size=32
        self.x=tf.placeholder(tf.int32,[self.num_batch,self.timestep])
        self.y=tf.placeholder(tf.int32,[self.num_batch,self.timestep])
        self.initstate=tf.zeros([self.num_batch,self.state_size])

        self.rnn_inputs=tf.one_hot(self.x,7)
        self.W=tf.Variable(tf.random.normal([self.state_size,7]))
        self.b=tf.Variable(tf.random.normal([7]))
        self.rnn_outputs,self.final_state=self.model()
        logits=tf.reshape(tf.matmul(tf.reshape(self.rnn_outputs,[-1,self.state_size]),self.W)+self.b,[self.num_batch,self.timestep,7])
        self.cost=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=logits)
        self.avgcost=tf.reduce_mean(self.cost)
        self.optimizer=tf.compat.v1.train.AdagradOptimizer(0.001).minimize(self.avgcost)

    def model(self):
        cell=tf.keras.layers.SimpleRNNCell(self.state_size)
        rnn_outputs,final_state=tf.nn.dynamic_rnn(cell,self.rnn_inputs,initial_state=self.initstate)
        return rnn_outputs,final_state

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            traincosts=[]
            d=Data()
            training_state=np.zeros((self.num_batch,self.state_size))
            for step,(X,Y) in enumerate(d.generate_epochs()):
                _,training_cost,training_state,_=sess.run([self.cost,self.avgcost,self.final_state,self.optimizer],feed_dict={self.x:X,self.y:Y,self.initstate:training_state})
                traincosts.append(training_cost)
            return traincosts


m=Model()

x1=np.linspace(1,413,600)
cost=m.train()

plt.plot(x1,cost)
plt.show()