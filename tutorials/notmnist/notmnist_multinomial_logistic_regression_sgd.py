""" 

Not Mnist assignment 1
from Udacity
Deep Learning course
Algorithm used: multinomial logistic regression
Optimizer used: Stochastic gradient descent
Own implementation

"""

import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt
import random as ran
import pickle

# Define paramaters for the model
learning_rate = 0.01
batch_size = 1000
n_epochs = 10


# Step 1: Read in data
# using pickle function to load notMNIST data from the folder data/notmnist
notmnist = pickle.load(open('/home/williamn/Repository/data/notmnist/notMNIST.pickle', "rb")) 

train_dataset = notmnist['train_dataset']
train_labels = notmnist['train_labels']
test_dataset = notmnist['test_dataset']
test_labels = notmnist['test_labels']

# Represent each image as 1x784 tensor and one hot encoded labels
train_dataset = train_dataset.reshape([200000,784])
test_dataset = test_dataset.reshape([test_dataset.shape[0],784])

tmp_train_labels = train_labels
tmp_test_labels = test_labels

# Labels convert from a fixed number to one hot encoding
#print(tmp_train_labels.size, tmp_train_labels.max())
train_labels = np.zeros((tmp_train_labels.size, tmp_train_labels.max() + 1))
train_labels[np.arange(tmp_train_labels.size), tmp_train_labels] = 1

test_labels = np.zeros((tmp_test_labels.size, tmp_test_labels.max() + 1))
test_labels[np.arange(tmp_test_labels.size), tmp_test_labels] = 1

# visualize data information
print('train dataset', train_dataset.shape, type(train_dataset))
print('train labels', train_labels.shape, type(train_labels))
print('test dataset', test_dataset.shape, type(test_dataset))
print('test labels', test_labels.shape, type(test_labels))

# visualize images (original data)
ran_image = ran.randint(0, train_dataset.shape[0])
print('Random label: ' + str(train_labels[ran_image]))
print('Random image: ')
#print(train_dataset[ran_image].reshape([1, 784]))
label = train_labels[ran_image].argmax(axis=0)
image = train_dataset[ran_image].reshape([28,28])
plt.title('Example: %d Label: %d' % (ran_image, label))
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()


# Step 2: create placeholders for features and labels
# each image in the notMNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to English letters a - j. 
X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y_placeholder')

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
logits = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)   # computes the mean  over examples in the batch


# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(train_dataset.shape[0]/batch_size)
	
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for j in range(n_batches):
			X_batch = train_dataset[j*batch_size:(j+1)*batch_size,]
			Y_batch = train_labels[j*batch_size:(j+1)*batch_size,]
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})

			total_loss += loss_batch
		print( 'Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print( 'Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	n_batches = int(test_dataset.shape[0]/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch = test_dataset[i*batch_size:(i+1)*batch_size,]
		Y_batch = test_labels[i*batch_size:(i+1)*batch_size,]
		#print(i)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch}) 
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)	
	
	print( 'Accuracy {0}'.format(total_correct_preds/test_dataset.shape[0]))

	# Visualization of the predicctions
	ran_image = ran.randint(0, X_batch.shape[0])
	print('Random test label {0}'.format(Y_batch[ran_image]))
	
	pred_label = tf.argmax(preds, 1)
	print('Predicted value = %d' % sess.run(pred_label[ran_image]))

	label = Y_batch[ran_image].argmax(axis=0)
	image = X_batch[ran_image].reshape([28,28])
	plt.title('Example test: %d Label: %d' % (ran_image, label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()

	writer.close()