from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *

class PixNet(object):
	def __init__(self, hr_images, lr_images, scope):
		self.inputs=lr_images		
		with tf.variable_scope(scope) as scope:
			self.train = tf.placeholder(tf.bool)
			self.construct_net(hr_images, lr_images)
			

	def softmax_loss(self, logits, labels):
		
		'''print(labels)

		# label_batch is a tensor of numeric labels to process
		# 0 <= label < num_labels

		new_labels=tf.one_hot(indices=logits.shape,depth=256)
		return tf.losses.softmax_cross_entropy(labels=new_labels,logits=logits)	'''	
		#print(labels.shape)
		#print(logits.shape)		
		logits = tf.reshape(logits, [-1, 256])
		#print(logits.shape)		
		labels = tf.cast(labels, tf.int32)
		labels = tf.reshape(labels, [-1])
		return tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
	
	'''
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o
	'''
	def _relu(self, x, name):
		return tf.nn.elu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
		with tf.variable_scope(name) as scope:
			o = tf.contrib.layers.batch_norm(
				x,
				scale=True,
				activation_fn=activation_fn,
				is_training=is_training,
				trainable=trainable,
				scope=scope)
			return o
	def build_adaptnet(self,inputs):
		outputs=maskconv2d(inputs,64,[3,3],[1,1],None,scope='adapt1')
		outputs=maskconv2d(outputs,64,[3,3],[1,1],None,scope='adapt2')
		outputs=maskconv2d(outputs,64,[3,3],[1,1],None,scope='adapt3')
		#outputs=self._relu(outputs,name='reluadapt')
		return outputs
	def build_pixelcnn(self,inputs):
		outputs=maskconv2d(inputs,64,[7,7],[1,1],'A',scope='pixelcnn_A_conv2d')
		inputs=outputs
		state=outputs
		for i in range(10):
			 inputs, state = gated_conv2d(inputs, state, [5, 5], scope='pixelcnn_gated' + str(i))
		outputs=maskconv2d(inputs,1024,[1,1],[1,1],'B',scope='pixelcnn_B1_conv2d')
		outputs=maskconv2d(outputs,256*3,[1,1],[1,1],'B',scope='pixelcnn_B2_conv2d')
		#outputs=self._relu(outputs,name='pixeladapt')		
		return outputs
	def construct_net(self, hr_images, lr_images):
		#labels
		labels = hr_images
		#normalization images [-0.5, 0.5]
		self.adapt_outputs=self.build_adaptnet(self.inputs)
		self.pixout=self.build_pixelcnn(self.adapt_outputs)
		loss = self.softmax_loss(self.pixout, labels)
		self.loss = tf.reduce_mean(loss)
		tf.summary.scalar('loss', self.loss)
		


