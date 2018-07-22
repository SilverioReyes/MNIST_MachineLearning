##################################################################
# Author:	        Silverio Reyes
# Filename:	        MNIST_ML_ComputerVisionDataSet.py
# Date Created:     07/21/2018
# Organization:     Oregon Institute of Technology
#
# Description:      MNIST is a simple computer vision dataset.
#                   it consists of images of handwritten digits
#                   and for Machine Learning (ML), we learn about
#                   MNIST as if it were "Hello World" for writing
#                   your first lines in programming.
#
# Task:             For this program, we will be training a model
#                   to look at images and predict what digits they
#                   are.
#
# Model:            Softmax Regression (multinomial logistic regression)
#
# Source:           
# https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners
##################################################################
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The computuer vision data set is split into three parts:
# 55,000 data points of training data (mnist.train)
# 10,000 data points of test data (mnist.test)
#  5,000 data points of validation data (mnist.validation)
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Basesd on the data points the image is composed by 28px by 28px which is a 784 dimensional vector
x = tf.placeholder(tf.float32, [None, 784])

# Need weights and biases for our computer vision data set model
# Each variable, respectfully is initialized as tensors full of zeros
# For variable W, the shape is 784 because we want to multiply the 784 dimensional image vector by it
# to produce 10 dimensional vectors of evidence for the different classes
# Variable b has a shape of 10 so we can add it to the output.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Will be using the softmax function. Think of it as exponentiating its inputs and then normalizing it
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Next we will be implementing cross-entropy which measures how inefficient 
# our prediction are for describing the truth
# Need placebolder to input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# Since we now know what we want our model to do, we need it to train. We will used
# backprogation to determine how the variables (y) affect the loss that we ask to minimize.
# This we can apply an optimization algorithm to modify the variables and reduce the loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# The learning rate given to the gradient descent is .5. Next we launch the model in an
# interactive session
session = tf.InteractiveSession()

# Initialize the variables created
tf.global_variables_initializer().run()

# Begin Training procedure for 1000 iterations
# Each step of the loop, we get a batch of 100 random data points from the training set
# Using small batches such ast (100) of random data is called stochastic training
# aka stochastic gradient descent. This is cheap and has same benefit oppose to sampling ever step
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluating our model
# Check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Convert he list of booleans to floating point numbers and take the mean
# Example: [True, False, True, True] after conversion equals [1, 0, 1, 1]. Mean = .75 (3/4)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Display the accuracy on the test data
# Should be about 92% ceiling, but needs improvement
# Capture log errors. This might be unnessesary
tf.logging.set_verbosity(old_v)
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




