# this file contains the main script to train the model and produce prediction from the 
# ICDM2018 competition test data

import argparse
import tensorflow as tf
import numpy as np
from models.baseline_model import ConvLSTM_Model
from config import config
from preprocess.data_reader import data_reader

def get_parser():
	parser = argparse.AugmentParser()
	parser.add_augment('--mode', type=str, choice=['train', 'test'])
	parser.add_augment('--data_file', type=str, help='the path to the tfrecords file')
	parser.add_augment('--batch_size', type=int, default=4, help='the batch_size')
	parser.add_augment('--save_interval', type=int, default=20)
	parser.add_augment('--log_interval', type=int, default=5)
	return parser

def train(args):
	'''
		this function is expected to be containing the main script to train the model 
	defined in model folder
	'''
	dr = data_reader(args.data_file)
	sess = tf.InteractiveSession()
	img, pred = dr.read_and_decode(args.batch_size)
	# img.shape == (2, 31, 501, 501, 1)
	# pred.shape == (2, 30, 501, 501, 1)
	# use the img and pred to build the graph and train the model
	model = ConvLSTM_Model(args)
	model.build(img, pred)
	model.fit(sess, img, pred)

	#model.predict(sess, test_data)

def test(args):
	'''
		this function is expected to be containing the script that takes in the test 
	data and make prediction(convert it into tfrecords format, in this way this op 
	can be build in the graph which can speed up)
	'''
	pass


def main():
	pass

if __name__ == '__main__':
	main()


