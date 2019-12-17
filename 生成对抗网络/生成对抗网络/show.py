
import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

import pickle

import 生成对抗网络

 

UNITS_SIZE = 生成对抗网络.UNITS_SIZE

 

def generatorImage(image_size):

    sample_images = tf.placeholder(tf.float32, [None, 100])

    G_logits, G_output = 生成对抗网络.generatorModel(sample_images, UNITS_SIZE, image_size)

    saver = tf.train.Saver()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        saver.restore(session, tf.train.latest_checkpoint('.'))

        sample_noise = np.random.uniform(-1, 1, size=(25, 100))

        samples = session.run(G_output, feed_dict={sample_images:sample_noise})

    with open('samples.pkl', 'wb') as f:

        pickle.dump(samples, f)

 

def show():

    with open('samples.pkl', 'rb') as f:

        samples = pickle.load(f)

    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)

    for ax, image in zip(axes.flatten(), samples):

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        ax.imshow(image.reshape((28, 28)), cmap='Greys_r')

    plt.show()

 

def main(argv=None):

    image_size = 生成对抗网络.mnist.train.images[0].shape[0]

    generatorImage(image_size)

    show()

 

if __name__ == '__main__':

    tf.app.run()

