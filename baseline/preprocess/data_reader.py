# this file contains script that build the data reading precedure into the graph
# for this project(TJCS Dev Group- pred-rain) the data has two parts, use the 
# previous 31 frames of radar map to produce prediction of the next 30 frame of radar map
# in the ICDM2018 competition only 6 frame is required to be evaluate (5, 10, 15, 20, 25, 30)

# Created by Tennant 2018-08-02
import tensorflow as tf
import numpy as np

class data_reader():
	def __init__(self, file_name):
		self.file_name = file_name

	def read_and_decode(self, batch_size, mode='train'):
		def parser_train(record):
			features = tf.parse_single_example(record, 
						features={'data_raw': tf.FixedLenFeature([], tf.string),
								'label_raw': tf.FixedLenFeature([], tf.string)})
			image = tf.decode_raw(features['data_raw'], tf.uint8)
			img = tf.reshape(image, [31, 501, 501, 1])
			img = tf.cast(img, tf.float32)
			label = tf.decode_raw(features['label_raw'], tf.uint8)
			label = tf.reshape(label, [30, 501, 501, 1])
			label = tf.cast(label, tf.float32)
			return img, label

		def parser_test(record):
			features = tf.parse_single_example(record, 
						features={'data_raw': tf.FixedLenFeature([], tf.string)})
			image = tf.decode_raw(features['data_raw'], tf.uint8)
			img = tf.reshape(image, [31, 501, 501, 1])
			img = tf.cast(img, tf.float32)
			return img
		
		dataset = tf.data.TFRecordDataset(self.file_name)
		if mode == 'train':
			dataset = dataset.map(parser_train).repeat().batch(batch_size)
		else:
			dataset = dataset.map(parser_test).repeat().batch(batch_size)

		iterator = dataset.make_one_shot_iterator()
		if mode == 'train':
			img_input, label = iterator.get_next()
			return img_input, label
		else:
			img_input = iterator.get_next()
			return img_input


	def read_data_in_np(self, sess, batch_size):
		pass

if __name__ == '__main__':
	dr = data_reader('01.tfrecords')
	sess = tf.InteractiveSession()
	img, lbl = dr.read_and_decode(10)
	# img and lbl will be tf.Tensor
	img_, lbl = sess.run([img, lbl])
	# use the Tensors to directly build the compute graph, no need for feed_dict
	print(img_.shape)
	print(lbl.shape)
