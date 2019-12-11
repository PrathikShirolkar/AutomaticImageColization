import tensorflow as tf
import sys
sys.path.insert(0, './')
from solver import *
import argparse
flags = tf.app.flags

#solver
flags.DEFINE_string("train_dir", "models", "trained model save path")
flags.DEFINE_string("test_dir", "testmodels", "trained model save path")
flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
flags.DEFINE_string("test_samples_dir", "test_samples", "sampled images save path")
flags.DEFINE_string("images_list_path", "train.txt", "train images list file path")
flags.DEFINE_string("test_images_list_path", "test.txt", "test images list file path")

#flags.DEFINE_string("chroma_list_path", "chroma.txt", "validation images list file path")

flags.DEFINE_string("model_dir","deepmodel","deeplab model directory")

flags.DEFINE_boolean("use_gpu", False, "whether to use gpu for training")
flags.DEFINE_integer("device_id", 0, "gpu device id")

flags.DEFINE_integer("num_epoch", 3000, "train epoch num")
flags.DEFINE_integer("batch_size", 8, "batch_size")

flags.DEFINE_float("learning_rate", 0.0003, "learning rate")

flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')

conf = flags.FLAGS

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		if args.option=='train':		
			solver = Solver(sess,'train')
		elif args.option=='test':
			solver = Solver(sess,'test')
		getattr(solver, args.option)()

if __name__ == '__main__':
	tf.app.run()

