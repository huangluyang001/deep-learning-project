import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class RBM():
    def __init__(self, n_input, n_hidden, alpha=1.0):
        self.n_input = n_input
        self.n_hidden = n_hidden

        # for tensorflow saving
        all_weights = dict()
        all_weights['w'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden], stddev=0.01, dtype=tf.float32),
                                       name='weight')
        all_weights['a'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='a')
        all_weights['b'] = tf.Variable(tf.random_uniform([self.n_hidden], dtype=tf.float32), name='b')

        self.weights = all_weights

        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.rbm_w = tf.placeholder(tf.float32, [self.n_input, self.n_hidden])
        self.rbm_a = tf.placeholder(tf.float32, [self.n_input])
        self.rbm_b = tf.placeholder(tf.float32, [self.n_hidden])

        # variables
        self.w_matrix = np.zeros([self.n_input, self.n_hidden], np.float32)
        self.a_vector = np.zeros([self.n_input], np.float32)
        self.b_vector = np.zeros([self.n_hidden], np.float32)
        self.out_w_matrix = np.random.normal(0.0, 0.01, [self.n_input, self.n_hidden])
        self.out_a_vector = np.zeros([self.n_input], np.float32)
        self.out_b_vector = np.zeros([self.n_hidden], np.float32)

        # One Gibbs Sample
        self.h_prob = tf.sigmoid(tf.matmul(self.x, self.rbm_w) + self.rbm_b)
        # sample hidden vector
        self.h = tf.nn.relu(tf.sign(self.h_prob - tf.random_uniform(tf.shape(self.h_prob))))
        self.v_new = tf.sigmoid(tf.matmul(self.h_prob, tf.transpose(self.rbm_w)) + self.rbm_a)
        self.h_new = tf.nn.sigmoid(tf.matmul(self.v_new, self.rbm_w) + self.rbm_b)

        # compute gradients
        self.w_grad_1 = tf.matmul(tf.transpose(self.x), self.h)
        self.w_grad_2 = tf.matmul(tf.transpose(self.v_new), self.h_new)

        # SGD
        self.update_w = self.rbm_w + alpha * (self.w_grad_1 - self.w_grad_2) / tf.to_float(tf.shape(self.x)[0])
        self.update_a = self.rbm_a + alpha * tf.reduce_mean(self.x - self.v_new, 0)
        self.update_b = self.rbm_b + alpha * tf.reduce_mean(self.h_prob - self.h_new, 0)

        # calculate errors
        self.h_sample = tf.nn.sigmoid(tf.matmul(self.x, self.rbm_w) + self.rbm_b)
        self.v_sample = tf.nn.sigmoid(tf.matmul(self.h_sample, tf.transpose(self.rbm_w)) + self.rbm_a)
        self.error = tf.reduce_mean(tf.square(self.x - self.v_sample))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def fit(self, batch_x):
        self.w_matrix, self.a_vector, self.b_vector = self.sess.run([self.update_w, self.update_a, self.update_b],
                                                                    feed_dict={self.x: batch_x, self.rbm_w:self.out_w_matrix, self.rbm_a: self.out_a_vector, self.rbm_b: self.out_b_vector})

        self.out_w_matrix = self.w_matrix
        self.out_a_vector = self.a_vector
        self.out_b_vector = self.b_vector

        return self.sess.run(self.error,  feed_dict={self.x: batch_x, self.rbm_w:self.out_w_matrix, self.rbm_a: self.out_a_vector, self.rbm_b: self.out_b_vector})


    def reconstruct(self, batch_x):
        return self.sess.run(self.v_sample, feed_dict={self.x: batch_x, self.rbm_w:self.out_w_matrix, self.rbm_a:self.out_a_vector, self.rbm_b:self.out_b_vector})

    def save_weights(self):

        self.sess.run(self.weights['w'].assign(self.out_w_matrix))

        self.sess.run(self.weights['vb'].assign(self.out_a_vector))

        self.sess.run(self.weights['hb'].assign(self.out_b_vector))

        saver = tf.train.Saver({'weight': self.weights['w'],

                                'a': self.weights['a'],

                                'b': self.weights['b']})

        save_path = saver.save(self.sess, 'rbm1.chp')

    def restore_weights(self):
        saver = tf.train.Saver({'weight': self.weights['w'],
                                'a': self.weights['a'],
                                'b': self.weights['b']})

        saver.restore(self.sess, 'rbm1.chp')

        self.out_w_matrix = self.weights['w'].eval(self.sess)
        self.out_a_vector = self.weights['a'].eval(self.sess)
        self.out_b_vector = self.weights['b'].eval(self.sess)


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    rbm = RBM(n_input=784, n_hidden=500, alpha=0.01)
    print('Initialized')

    for j in range(10):
        for i in range(500):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            error = rbm.fit(batch_xs)
        print('Batch:',j)
        print('Error:',error)
        print('------------------------')


    def show_digit(x):
        plt.imshow(x.reshape((28, 28)),cmap ='gray')
        plt.show()


    mnist_images = mnist.test.images[:1]
    for mnist_image in mnist_images:
        image_initial = mnist_image.reshape(1,-1)
        image_rec = rbm.reconstruct(image_initial)
        show_digit(image_initial)
        show_digit(image_rec)