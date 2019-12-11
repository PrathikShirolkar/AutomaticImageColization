import tensorflow as tf
import numpy as np
from ops import *
from pixnet import *
from network import *
from data import *
from utils import *
from refinement import *
import os
import time
flags=tf.app.flags
conf=flags.FLAGS
class Solver(object):
	def __init__(self,sess,option):
		self.option=option		
		if option=='train':
			self.device_id = conf.device_id
			self.sess=sess
			self.train_dir = conf.train_dir
			self.samples_dir = conf.samples_dir
			if not os.path.exists(self.train_dir):
				os.makedirs(self.train_dir)
			if not os.path.exists(self.samples_dir):
				os.makedirs(self.samples_dir)    
			#datasets params
			self.num_epoch = conf.num_epoch
			self.batch_size = conf.batch_size
			#optimizer parameter
			self.learning_rate = conf.learning_rate
			if conf.use_gpu:
				device_str = '/gpu:' +  str(self.device_id)
			else:
				device_str = '/cpu:0'
			with tf.device(device_str):
				self.dataset=DataSet(conf.images_list_path,self.num_epoch,self.batch_size)
				self.condnet=Deeplab_v2(self.dataset.lr_images,21,False)
				self.loader=tf.train.Saver(var_list=tf.global_variables())
				self.sess.run(tf.global_variables_initializer())
				self.sess.run(tf.local_variables_initializer())
				checkpoint=conf.model_dir + '/deeplab_resnet_init.ckpt'
				self.load(self.loader,checkpoint)
				self.pixelnet=PixNet(self.dataset.hr_images,self.condnet.deeplab_temp_outputs,'prsr')
				self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
				#learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,500000, 0.5, staircase=True)
				optimizer=tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, epsilon=1e-8)
				self.train_op = optimizer.minimize(self.pixelnet.loss, global_step=self.global_step)
							
				'''var_list=tf.trainable_variables()	
				variableslist= [v for v in var_list if 'pixelcnn' in v.name or 'adapt' in v.name or '2a' in v.name or '2b' in v.name or '2c' in v.name or '3a' in v.name or '3b' in v.name or '4a' in v.name or '4b' in v.name or '5a' in v.name or '5b' in v.name or '5c' in v.name ]
				finallist=[v for v in variableslist if 'weights' in v.name or 'biases' in v.name]		
				optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.9, epsilon=1e-8)
				grads = optimizer.compute_gradients(self.pixelnet.loss, var_list=var_list)
				self.train_op=optimizer.apply_gradients(grads)'''
				
		elif option=='test':	
			self.device_id = conf.device_id
			self.sess=sess
			self.train_dir = conf.train_dir
			self.samples_dir = conf.samples_dir
			self.num_epoch = 1000
			self.batch_size = 1
			self.test_samples_dir = conf.test_samples_dir
			if conf.use_gpu:
				device_str = '/gpu:' +  str(self.device_id)
			else:
				device_str = '/cpu:0'
			with tf.device(device_str):
				self.dataset=DataSet(conf.test_images_list_path,self.num_epoch,self.batch_size,option='test')
				self.condnet=Deeplab_v2(self.dataset.lr_images,21,False)
				#self.loader=tf.train.Saver(var_list=tf.global_variables())
				self.sess.run(tf.global_variables_initializer())
				self.sess.run(tf.local_variables_initializer())
				checkpoint=conf.model_dir + '/deeplab_resnet_init.ckpt'
				#self.load(self.loader,checkpoint)
				self.pixelnet=PixNet(self.dataset.hr_images,self.condnet.deeplab_temp_outputs,'prsr')				
				#var_list=tf.trainable_variables()	
				#variableslist= [v for v in var_list if 'pixelcnn' in v.name or 'adapt' in v.name]
				#finallist=[v for v in variableslist if 'weights' in v.name or 'biases' in v.name]		
				self.loader1=tf.train.Saver(var_list=tf.global_variables())
				self.sess.run(tf.global_variables_initializer())
				self.sess.run(tf.local_variables_initializer())
				checkpoint=conf.model_dir + '/model.ckpt-210000'
				self.load(self.loader1,checkpoint)
				
	def train(self):
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver()
		# Create a session for running operations in the Graph.
		#config = tf.ConfigProto(allow_soft_placement=True)
		#config.gpu_options.allow_growth = True
		#sess = tf.Session(config=config)
		# Initialize the variables (like the epoch counter).
		self.sess.run(init_op)
		summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
		iters = 0
		try:
			while not coord.should_stop():
				t1 = time.time()
				_, loss = self.sess.run([self.train_op, self.pixelnet.loss], feed_dict={self.pixelnet.train: True})
				t2 = time.time()
				print('step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
				iters += 1
				if iters % 10 == 0:
					summary_str = self.sess.run(summary_op, feed_dict={self.pixelnet.train: True})
					summary_writer.add_summary(summary_str, iters)
				if iters % 1000 == 0:
					self.sample(mu=1.1, step=iters)
				if iters % 10000 == 0:
					checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
					saver.save(self.sess, checkpoint_path, global_step=iters)
		except tf.errors.OutOfRangeError:
			checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
			saver.save(self.sess, checkpoint_path)
			print('Done training -- epoch limit reached')
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
		coord.join(threads)
		self.sess.close()
	
	def test(self):
		#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		saver = tf.train.Saver()
		# Create a session for running operations in the Graph.
		#config = tf.ConfigProto(allow_soft_placement=True)
		#config.gpu_options.allow_growth = True
		#sess = tf.Session(config=config)
		# Initialize the variables (like the epoch counter).
		#self.sess.run(init_op)
		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
		iters = 0
		try:
			while not coord.should_stop():
				self.sample(mu=1.1, step=iters)
				iters+=1
		except tf.errors.OutOfRangeError:
			print('Done testing')
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
		coord.join(threads)
		self.sess.close()

	def save(self, saver, step):
		'''
		Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')

	def load(self, saver, filename):
		'''
		Load trained weights.
		''' 
		saver.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))
		
	def sample(self, mu=1.1, step=None):
		if self.option=='train':
			c_logits = self.pixelnet.pixout
			lr_imgs = self.dataset.lr_images
			hr_imgs = self.dataset.hr_images
			np_hr_imgs, np_lr_imgs = self.sess.run([hr_imgs, lr_imgs])
			gen_hr_imgs = np.zeros((self.batch_size, 28, 28, 3), dtype=np.float32)
			np_c_logits = self.sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.pixelnet.train:False})
			print('iters %d: ' % step)
			for i in range(28):
				for j in range(28):
					for c in range(3):
						new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=mu)
						gen_hr_imgs[:, i, j, c] = new_pixel
    		
			save_samples(np_lr_imgs, self.samples_dir + '/lr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(np_hr_imgs, self.samples_dir + '/hr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(gen_hr_imgs, self.samples_dir + '/generate_' + str(mu*10) + '_' + str(step) + '.jpg')

		elif self.option=='test':
			c_logits = self.pixelnet.pixout
			lr_imgs = self.dataset.lr_images
			hr_imgs = self.dataset.hr_images
			np_hr_imgs, np_lr_imgs = self.sess.run([hr_imgs, lr_imgs])
			gen_hr_imgs = np.zeros((self.batch_size, 28, 28, 3), dtype=np.float32)
			np_c_logits = self.sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.pixelnet.train:False})
			#print(np_c_logits)			
			#print('iters %d: ' % step)
			for i in range(28):
				for j in range(28):
					for c in range(3):
						new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=mu)
						gen_hr_imgs[:, i, j, c] = new_pixel
    		
			save_samples(np_lr_imgs, self.test_samples_dir + '/lr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(np_hr_imgs, self.test_samples_dir + '/hr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(gen_hr_imgs, self.test_samples_dir + '/generate_' + str(mu*10) + '_' + str(step) + '.jpg')
			
