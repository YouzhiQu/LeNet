
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet
import numpy as np
import matplotlib.pyplot as plt

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER


    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.initialize_all_variables())

    temp_step = 10
    result_step = np.arange(0, temp_step*100, 100)
    result_acc = np.zeros(temp_step)
    result_loss = np.zeros(temp_step)
    result_test = np.zeros(temp_step)
    for i in range(temp_step*100+1):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            r = int(i / 100 - 1)
            result_acc[r] = sess.run(lenet.train_accuracy,feed_dict={
                lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]
            })
            result_loss[r] = sess.run(lenet.loss, feed_dict={
                lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g, training loss %g" % (i, result_acc[r], result_loss[r]))
            result_test[r] = sess.run(lenet.train_accuracy, feed_dict={
                lenet.raw_input_image: mnist.test.images, lenet.raw_input_label: mnist.test.labels
            })
            print("test accuracy %g" % (result_test[r]))

        sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
    save_path = saver.save(sess, parameter_path)
    plt.plot(result_step, result_acc, label='training accuracy')
    plt.plot(result_step, result_test, label='test accuracy')
    plt.title('LeNet')
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()


