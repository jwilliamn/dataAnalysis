""" 

Not Mnist assignment 3
from Udacity
Deep Learning course
Task: 1. Introduce and tune L2 regularization.
	  2. Demonstrate an extreme case of overfitting.
	  3. Introduce dropout on the hidden layer.
	  4. Try to get the best performance over 97.1%
Algorithm used: 1-hidden layer neural network
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
learning_rate = 0.01 # 0.01
batch_size = 100  # 100 - 10000) Task 2. Restrict training data to a few batches
n_epochs = 15

# Network Parameters
n_hidden = 1024  # hidden layer number of features
n_input = 784  # Not Mnist data input (img shape: 28*28)
n_classes = 10  # Not Mnist total classes (a - j)

# Regularization
beta = 0.01

# Step 1: Read in data
pickle_file = '/home/williamn/Repository/data/notmnist/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
	notmnist = pickle.load(f)
	train_dataset = notmnist['train_dataset']
	train_labels = notmnist['train_labels']
	valid_dataset = notmnist['valid_dataset']
	valid_labels = notmnist['valid_labels']
	test_dataset = notmnist['test_dataset']
	test_labels = notmnist['test_labels']
	del notmnist  # hint to help gc free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

# Reformat into a shape that's more adapted to the model
image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape, type(train_dataset))
print('Validation set', valid_dataset.shape, valid_labels.shape, type(valid_dataset))
print('Test set', test_dataset.shape, test_labels.shape, type(test_dataset))

# visualize a random image (original data)
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

keep_prob = tf.placeholder(tf.float32) # probability that a neuron's output is kept during
									   # dropout

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and the next layer (i.e so that Y = X * w + b)
# shape of b depends on the next layers (i.e Y)
weights = {
	'w1' : tf.Variable(tf.random_normal(shape=[n_input, n_hidden], stddev=0.01), name="weights_1"),
	'w2' : tf.Variable(tf.random_normal(shape=[n_hidden, n_classes], stddev=0.01), name="weights_2")
}

biases = {
	'b1' : tf.Variable(tf.zeros([1, n_hidden]), name="bias_1"),
	'b2' : tf.Variable(tf.zeros([1, n_classes]), name='bias_2')
}


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
def one_hidden_layer_nn(X, weights, biases, keep_prob):
	# Hidden layer with RELU activation
	h_layer = tf.matmul(X, weights['w1']) + biases['b1']
	h_layer = tf.nn.relu(h_layer)

	h_layer_drop = tf.nn.dropout(h_layer, keep_prob)  # Task 3. Introduce dropout
	#o_layer = tf.matmul(h_layer, weights['w2']) + biases['b2']
	o_layer = tf.matmul(h_layer_drop, weights['w2']) + biases['b2']
	return o_layer

logits = one_hidden_layer_nn(X, weights, biases, keep_prob)



# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)   # computes the mean  over examples in the batch
loss = (loss + 
		beta*tf.nn.l2_loss(weights['w1']) + 
		beta*tf.nn.l2_loss(weights['w2'])) # task 1. Introducing regularization


# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

## Valid and test prediction
#valid_prediction = tf.nn.softmax(
#tf.matmul(tf_valid_dataset, weights) + biases)
#test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/one_hidden_layer_regularization_nn', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(train_dataset.shape[0]/batch_size)
	
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for j in range(n_batches):
			X_batch = train_dataset[j*batch_size:(j+1)*batch_size,]
			Y_batch = train_labels[j*batch_size:(j+1)*batch_size,]
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch, keep_prob: 0.5})

			total_loss += loss_batch
		print( 'Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print( 'Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	print('Parameters: ', sess.run(weights), sess.run(biases))


	trained_weights = sess.run(weights)
	trained_biases = sess.run(biases)

	# Model that takes trained weights and biases for validation and test prediction
	def one_hidden_layer_no_dropout_nn(X, weights, biases):
		# Hidden layer with RELU activation
		h_layer = tf.matmul(X, weights['w1']) + biases['b1']
		h_layer = tf.nn.relu(h_layer)
		o_layer = tf.matmul(h_layer, weights['w2']) + biases['b2']
		return o_layer

	logitsValid = one_hidden_layer_no_dropout_nn(X, trained_weights, trained_biases)

	# test the model
	n_batches = int(test_dataset.shape[0]/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch = valid_dataset[i*batch_size:(i+1)*batch_size,]
		Y_batch = valid_labels[i*batch_size:(i+1)*batch_size,]
		#print(i)
		logits_batch = sess.run(logitsValid, feed_dict={X: X_batch, Y:Y_batch}) 
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)	
	
	print( 'Accuracy {0}'.format(total_correct_preds/test_dataset.shape[0]))

	# Visualization of the predicctions
	for _ in range(10):
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