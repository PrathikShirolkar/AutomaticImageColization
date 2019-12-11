from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

class DataSet(object):
	def read_labeled_image_list(self,image_list_file):
		#print(image_list_file)
		"""Reads a .txt file containing pathes and labeles
		Args:
			image_list_file: a .txt file with one /path/to/image per line
			label: optionally, if set label will be pasted after each line
		Returns:
		List with all filenames in file image_list_file
		"""
		f = open(image_list_file, 'r')
		filenames = []
		labels = []
		for line in f:
			f, label = line[:-1].split(' ')
			filenames.append(f)
			labels.append(label)
		return filenames, labels
	def read_images_from_disk(self,input_queue):
		"""Consumes a single filename and label as a ' '-delimited string.
		Args:
			filename_and_label_tensor: A scalar string tensor.
		Returns:
			Two tensors: the decoded image, and the string label.
		"""
		flabel = tf.read_file(input_queue[1])
		file_contents = tf.read_file(input_queue[0])
		example = tf.image.decode_jpeg(file_contents,3)
		label=tf.image.decode_jpeg(flabel,3)		
		return example, label

	def __init__(self,filename,num_epochs,batch_size,option='train'):
		if option=='train':
			self.filename=filename
			self.num_epochs=num_epochs
			self.batch_size=batch_size
			image_list, label_list = self.read_labeled_image_list(image_list_file=self.filename)
		
			images = tf.convert_to_tensor(image_list, dtype=tf.string)
			labels = tf.convert_to_tensor(label_list, dtype=tf.string)
			
			input_queue = tf.train.slice_input_producer([images, labels],num_epochs=self.num_epochs,shuffle=True)
		
			image, label = self.read_images_from_disk(input_queue)
			#image = tf.image.decode_jpeg(image, 3)
			#label = tf.image.decode_jpeg(label, 3)
	
			hr_image = tf.image.resize_images(label, [28, 28])
			lr_image = tf.image.resize_images(image, [224, 224])
			hr_image = tf.cast(hr_image, tf.float32)
			lr_image = tf.cast(lr_image, tf.float32)
	
			min_after_dequeue = 1000
			capacity = min_after_dequeue + 400 * batch_size
			self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
		
		elif option=='test':
			self.filename=filename
			self.num_epochs=1
			self.batch_size=1
			image_list, label_list = self.read_labeled_image_list(image_list_file=self.filename)
	
			images = tf.convert_to_tensor(image_list, dtype=tf.string)
			labels = tf.convert_to_tensor(label_list, dtype=tf.string)
		
			input_queue = tf.train.slice_input_producer([images, labels],num_epochs=self.num_epochs,shuffle=True)

			image, label = self.read_images_from_disk(input_queue)
			#image = tf.image.decode_jpeg(image, 3)
			#label = tf.image.decode_jpeg(label, 3)
	
			hr_image = tf.image.resize_images(label, [28, 28])
			lr_image = tf.image.resize_images(image, [224, 224])
			hrr_image= tf.image.resize_images(label, [224,224])
			hr_image = tf.cast(hr_image, tf.float32)
			lr_image = tf.cast(lr_image, tf.float32)
			hrr_image = tf.cast(hrr_image, tf.float32)
	
			min_after_dequeue = 1000
			capacity = min_after_dequeue + 400 * batch_size
			self.hr_images, self.lr_images, self.hrr_images = tf.train.shuffle_batch([hr_image, lr_image, hrr_image], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
