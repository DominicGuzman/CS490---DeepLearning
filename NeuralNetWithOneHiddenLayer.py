import numpy as np
import math
from random import *
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data


def main():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.test.next_batch(10000)

    xTensor = tf.placeholder(tf.float32, [None, 784])
    yTensor = tf.placeholder(tf.float32, [None, 10])

    wHiddenLayer1 = tf.Variable(tf.random_normal([784, 512]))
    bHiddenLayer1 = tf.Variable(tf.random_normal([1, 512]))

    wOutputLayer = tf.Variable(tf.random_normal([512, 10]))
    bOutputLayer = tf.Variable(tf.random_normal([1, 10]))

    outputOfFirst = tf.matmul(xTensor, wHiddenLayer1) + bHiddenLayer1
    outputOfFirst = tf.nn.relu(outputOfFirst)
    

    yHat = tf.matmul(outputOfFirst, wOutputLayer) + bOutputLayer

    yHatSoftmax = tf.nn.softmax(yHat)

    lastError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yHat, labels=yTensor))

    optimizer = tf.train.AdamOptimizer(.15).minimize(lastError)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        t1 = sess.run(outputOfFirst, feed_dict={xTensor: Xtrain})

        #print(t1)

        for el in range(20):
            for el2 in range(50):
                sess.run(optimizer, feed_dict={xTensor: Xtrain[el*1000:(el+1)*1000],
                                               yTensor: Ytrain[el*1000:(el+1)*1000]})
            #print("This is from your iterations", el)

        resultTF = sess.run(yHatSoftmax, feed_dict={xTensor: Xtest[:10000]})


        indexOfHighest = np.argmax(resultTF, axis=1)
        indexOfActual = np.argmax(Ytest[0:10000], axis=1)

        numOfCorrect = 0
        for idx, el in enumerate(indexOfHighest):
            #print("This is the result", resultTF[idx])

            if el == indexOfActual[idx]:
                numOfCorrect = numOfCorrect + 1
            #print("These are the number of correct", el, indexOfActual[idx])

        actual = np.argmax(Ytest[:10000], axis=1)
        result = np.argmax(resultTF[:10000], axis=1)

        conf_result = confusion_matrix(y_true=actual,
                                       y_pred=result)

        currDiagonal = np.diagonal(conf_result)
        print("This is the sum of the diagonal", np.sum(currDiagonal))
        print("This is the sum of the diagonal / 10000", np.sum(currDiagonal) / 10000)
        print(conf_result)

        print("These are the number of correct classifications", numOfCorrect/len(Xtest))

    return

main()