import tensorflow as tf
import numpy as np

class data_reader():
	def __init__(self, file_name):
		self.file_name = file_name

	def read_and_decode(self, batch_size):
		def parser(record):
			features = tf.parse_single_example(record, 
						features={'data_raw': tf.FixedLenFeature([], tf.string),
								'label_raw': tf.FixedLenFeature([], tf.string)})
			image = tf.decode_raw(features['data_raw'], tf.uint8)
			img = tf.reshape(image, [31, 501, 501, 1])
			img = tf.cast(img, tf.float32)
			label = tf.decode_raw(features['label_raw'], tf.uint8)
			label = tf.reshape(label, [30, 501, 501, 1])
			return img, label
	
		dataset = tf.data.TFRecordDataset(self.file_name)
		
		dataset = dataset.map(parser).repeat().batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		img_input, label = iterator.get_next()
		return img_input, label

	def read_data_in_np(self, sess, batch_size):
		pass

#img, lbl = read_and_decode([data_path], 10)
#img_, lbl_ = sess.run([img, lbl])

if __name__ == '__main__':
	dr = data_reader('01.tfrecords')
	sess = tf.InteractiveSession()
	img, lbl = dr.read_and_decode(10)
	img_, lbl = sess.run([img, lbl])
	print(img_.shape)
	print(lbl.shape)
