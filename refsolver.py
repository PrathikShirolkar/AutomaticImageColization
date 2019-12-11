import tensorflow as tf
import numpy as np
from ops import *
from refinement import *
from refdata import *
from utils import *
import os
import time
flags=tf.app.flags
conf=flags.FLAGS
class RefSolver(object):
	def __init__(self,sess,option):
		self.option=option
		if(self.option=='train'):
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
				self.dataset=RefDataSet(conf.images_list_path,self.num_epoch,self.batch_size,option='train')
				self.refout=Refinement(self.dataset.lr_images,self.dataset.hr_images,self.dataset.gr_images,'ref')			
				self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
				optimizer=tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, epsilon=1e-8)
				self.train_op = optimizer.minimize(self.refout.loss, global_step=self.global_step)
		elif(self.option=='test'):
			self.device_id=conf.device_id
			self.sess=sess
			self.train_dir=conf.train_dir
			self.samples_dir = conf.samples_dir
			self.num_epoch = 1
			self.batch_size = 1
			self.test_samples_dir = conf.test_samples_dir
			if conf.use_gpu:
				device_str = '/gpu:' +  str(self.device_id)
			else:
				device_str = '/cpu:0'
			with tf.device(device_str):
				self.dataset=RefDataSet(conf.test_images_list_path,self.num_epoch,self.batch_size,option='test')
				self.refout=Refinement(self.dataset.lr_images,self.dataset.hr_images,self.dataset.gr_images,'ref')
				self.loader1=tf.train.Saver(var_list=tf.global_variables())
				self.sess.run(tf.global_variables_initializer())
				self.sess.run(tf.local_variables_initializer())
				checkpoint=conf.model_dir + '/model.ckpt-14000'
				self.load(self.loader1,checkpoint)
						
	def train(self):
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver()
		self.sess.run(init_op)
		summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
		iters = 0
		try:
			while not coord.should_stop():
				t1 = time.time()
				_, loss = self.sess.run([self.train_op, self.refout.loss], feed_dict={self.refout.train: True})
				t2 = time.time()
				print('step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
				iters += 1
				if iters % 10 == 0:
					summary_str = self.sess.run(summary_op, feed_dict={self.refout.train: True})
					summary_writer.add_summary(summary_str, iters)
				if iters % 1000 == 0:
					self.sample(mu=1.1, step=iters)
				if iters % 1000 == 0:
					checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
					saver.save(self.sess, checkpoint_path, global_step=iters)
		except tf.errors.OutOfRangeError:
			checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
			saver.save(self.sess, checkpoint_path)
			print('Done training -- epoch limit reached')
		finally:
			coord.request_stop()
		coord.join(threads)
		self.sess.close()
	
	def test(self):
		saver = tf.train.Saver()
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
		if(self.option=='train'):
			c_logits = self.refout.refoutputs
			lr_imgs = self.dataset.lr_images
			hr_imgs = self.dataset.hr_images
			gr_imgs = self.dataset.gr_images
			np_hr_imgs, np_lr_imgs,  np_gr_imgs = self.sess.run([hr_imgs, lr_imgs, gr_imgs])
			gen_hr_imgs = np.zeros((self.batch_size, 224, 224, 3), dtype=np.float32)
			np_c_logits = self.sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.refout.train:False})
			print('iters %d: ' % step)
			for i in range(224):
				for j in range(224):
					for c in range(3):
						new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=mu)
						gen_hr_imgs[:, i, j, c] = new_pixel
    		
			save_samples(np_lr_imgs, self.samples_dir + '/lr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(np_hr_imgs, self.samples_dir + '/hr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(gen_hr_imgs, self.samples_dir + '/generate_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(np_gr_imgs, self.samples_dir + '/gray_' + str(mu*10) + '_' + str(step) + '.jpg')			
		elif(self.option=='test'):
			c_logits = self.refout.refoutputs
			lr_imgs = self.dataset.lr_images
			hr_imgs = self.dataset.hr_images
			gr_imgs = self.dataset.gr_images
			np_hr_imgs, np_lr_imgs, np_gr_imgs = self.sess.run([hr_imgs, lr_imgs,gr_imgs])
			gen_hr_imgs = np.zeros((self.batch_size, 224, 224, 3), dtype=np.float32)
			np_c_logits = self.sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.refout.train:False})
			print('iters %d: ' % step)
			for i in range(224):
				for j in range(224):
					for c in range(3):
						new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=mu)
						gen_hr_imgs[:, i, j, c] = new_pixel
    		
			save_samples(np_lr_imgs, self.test_samples_dir + '/lr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(np_hr_imgs, self.test_samples_dir + '/hr_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(gen_hr_imgs, self.test_samples_dir + '/generate_' + str(mu*10) + '_' + str(step) + '.jpg')
			save_samples(np_gr_imgs, self.test_samples_dir + '/gray_' + str(mu*10) + '_' + str(step) + '.jpg')
			
