# this file contains the main script to train the model and produce prediction from the 
# ICDM2018 competition test data

import argparse
import tensorflow as tf
import numpy as np
from models.baseline_model import ConvLSTM_Model
from preprocess.data_reader import np_data_reader

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	parser.add_argument('--data_dir', type=str, default='data/', help='the path to the datas')
	parser.add_argument('--test_dir', type=str, help='the path to test data file')
	parser.add_argument('--model_path', default='pretrained/', type=str)
	parser.add_argument('--batch_size', type=int, default=1, help='the batch_size')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--save_interval', type=int, default=10)
	parser.add_argument('--max_to_keep', type=int, default=5, help='how many model to save')
	parser.add_argument('--log_interval', type=int, default=1)
	return parser

def train(args):
	'''
		this function is expected to be containing the main script to train the model 
	defined in model folder
	'''
	dr = np_data_reader(args)
	sess = tf.InteractiveSession()
	#img, pred = dr.read_and_decode(args.batch_size)
	# use the img and pred to build the graph and train the model
	model = ConvLSTM_Model(args)
	X = tf.placeholder(shape=[None, 31, 501, 501, 1], dtype=tf.float32)
	y = tf.placeholder(shape=[None, 30, 501, 501, 1], dtype=tf.float32)
	model.build(X, y)

	model.fit(sess, dr, args.model_path)

	#model.predict(sess, test_data)

def test(args):
	'''
		this function is expected to be containing the script that takes in the test 
	data and make prediction(convert it into tfrecords format, in this way this op 
	can be build in the graph which can speed up)
	'''
	raise NotImplementedError('Not implemented')
	dr = np_data_reader(args.test_data)
	sess = tf.InteractiveSession()
	model = ConvLSTM_Model(args)
	model.build(img)
	model.restore(args.model_path)
	# TODO: collect the result
	model.predict(img)

def main(args):
	if args.mode == 'train':
		train(args)
	else:
		test(args)

if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()
	main(args)


