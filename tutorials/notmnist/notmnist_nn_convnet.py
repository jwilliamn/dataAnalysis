""" 

Not Mnist assignment 4
from Udacity
Deep Learning course
Task: 1. Build a convnet model using maxpooling operation of stride size 2 and kernel size 2.
	  2. Try to get the best performance using convnet.
Algorithm used: 2 convolutional layers neural network
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
learning_rate = 0.05 # 0.01
batch_size = 16  # 100
patch_size = 5
n_epochs = 15

# Network Parameters
depth = 16
n_hidden = 64  # hidden layer number of features
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
num_channels = 1 # grayscale

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
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
X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, num_channels], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, n_classes], name='Y_placeholder')
X_val = tf.constant(valid_dataset)
X_test = tf.constant(test_dataset)


# Step 3: create weights and bias
# weights and biases are random initialized
# shape of w depends on the dimension of X and the next layer (i.e so that Y = X * w + b)
# shape of b depends on the next layers (i.e Y)
weights = {
	'w1' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, num_channels, depth], stddev=0.1)),
	'w2' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, depth, depth], stddev=0.1)),
	'w3' : tf.Variable(tf.random_normal(shape=[image_size // 4 * image_size // 4 * depth, n_hidden], stddev=0.1)),
	'w4' : tf.Variable(tf.random_normal(shape=[n_hidden, n_classes], stddev=0.1))
}

biases = {
	'b1' : tf.Variable(tf.zeros([depth])),
	'b2' : tf.Variable(tf.constant(1.0, shape=[depth])),
	'b3' : tf.Variable(tf.constant(1.0, shape=[n_hidden])),
	'b4' : tf.Variable(tf.constant(1.0, shape=[n_classes]))
}


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
def convnet_model(X):
	conv = tf.nn.conv2d(X, weights['w1'], [1, 2, 2, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b1'])
	conv = tf.nn.conv2d(hidden, weights['w2'], [1, 2, 2, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b2'])
	shape = hidden.get_shape().as_list()
	reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
	hidden = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])
	o_layer = tf.matmul(hidden, weights['w4']) + biases['b4']
	return o_layer

logits = convnet_model(X)


# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)


# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# Accuracy function
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


# Step 7: Prediction*
train_pred = tf.nn.softmax(logits)
valid_pred = tf.nn.softmax(convnet_model(X_val))
test_pred = tf.nn.softmax(convnet_model(X_test))

with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(train_dataset.shape[0]/batch_size)

	print('Initialized!', n_batches)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for j in range(n_batches):
			offset = (j * batch_size) % (train_labels.shape[0] - batch_size)
			X_batch = train_dataset[offset:(offset + batch_size),:,:,:]
			Y_batch = train_labels[offset:(offset + batch_size),:]
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch, preds = sess.run([optimizer, loss, train_pred], feed_dict={X:X_batch, Y:Y_batch})

			total_loss += loss_batch
			if(j % 1000 == 0):
				print('		Minibatch loss at step %d: %f' % (j, loss_batch))
				print('		Minibatch accuracy: %.1f%%' % accuracy(preds, Y_batch))
				print('		Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels))
		print('	Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels))
	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	print('Parameters: ', sess.run(weights), sess.run(biases))


	# Visualization of the predicctions on test data
	for _ in range(10):
		ran_image = ran.randint(0, test_dataset.shape[0])
		print('Random test label {0}'.format(test_labels[ran_image]))
		
		preds_ = test_pred.eval()
		pred_label = tf.argmax(preds_, 1)
		print('Predicted value = %d' % sess.run(pred_label[ran_image]))

		label = test_labels[ran_image].argmax(axis=0)
		image = test_dataset[ran_image].reshape([28,28])
		plt.title('Example test: %d Label: %d' % (ran_image, label))
		plt.imshow(image, cmap=plt.get_cmap('gray_r'))
		plt.show()

	writer.close()