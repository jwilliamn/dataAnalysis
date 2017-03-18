import tensorflow as tf
import numpy as np
import time
import os

import matplotlib.pyplot as plt
import random as ran
import pickle

from scipy import ndimage
from scipy.misc.pilutil import imread
import cv2


# Loading data in a more manageable format.
image_size = 128   # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

folder = '/home/williamn/Downloads/test/'



def load_letter(folder, min_num_images):
	""" Load the data for a single letter label. """
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype = np.float32)
	print(folder)
	num_images = 0
	for image in image_files:
		image_file = os.path.join(folder, image)

		print('image_file', image_file)
		try:
			#image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
			image_data = imread(image_file, flatten=True)
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			#print('image shape', image_data.shape)
			#dataset[num_images, :, :] = image_data
			num_images = num_images + 1
			#print('image type',type(image))

			
			plt.title('Example test: %s Label: %d' % (image, 13))
			plt.imshow(image_data, cmap='gray') #, cmap=plt.cm.gray)
			plt.show()

		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

	np.set_printoptions(threshold=np.nan)
	print(image_data)

	dataset = dataset[0:num_images, :, :]
	if num_images < min_num_images:
		raise Exception('many fewer images than expected: %d < %d' % (num_images, min_num_images))

	print('Full dataset tensor:', dataset.shape)
	return dataset


if __name__ == '__main__':
	dataset = load_letter(folder, 12)
	#plt.imshow(dataset[1])
	#plt.show()