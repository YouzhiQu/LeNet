
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import os
import lenet
from lenet import Lenet


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

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(lenet.train_accuracy,feed_dict={
                lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]
            })
            train_loss = sess.run(lenet.loss, feed_dict={
                lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
        sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
    save_path = saver.save(sess, parameter_path)
    test_accuracy = sess.run(lenet.train_accuracy, feed_dict={
        lenet.raw_input_image: mnist.test.images, lenet.raw_input_label: mnist.test.labels
    })
    print("test accuracy %g" % (test_accuracy))
if __name__ == '__main__':
    main()


