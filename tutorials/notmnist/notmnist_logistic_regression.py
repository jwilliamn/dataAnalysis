""" Not Mnist assignment 1
	from Udacity
	Deep Learning course
	Learning with logistic regression"""

import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt
import random as ran
import pickle

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
notmnist = pickle.load(open('/home/williamn/Repository/data/notmnist/notMNIST.pickle', "rb")) 

train_dataset = notmnist['train_dataset']
train_labels = notmnist['train_labels']
test_dataset = notmnist['test_dataset']
test_labels = notmnist['test_labels']

# visualize data information
print('train dataset', train_dataset.shape, type(train_dataset))
print('train labels', train_labels.shape, type(train_labels))
print('test dataset', test_dataset.shape, type(test_dataset))
print('test labels', test_labels.shape, type(test_labels))

# visualize images (original data)
ran_image = ran.randint(0, train_dataset.shape[0])
print('Random label: ' + str(train_labels[ran_image]))
print('Random image: ')
print(train_dataset[ran_image].reshape([1, 784]))
label = train_labels[ran_image]
image = train_dataset[ran_image]
plt.title('Example: %d Label: %d' % (ran_image, label))
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
