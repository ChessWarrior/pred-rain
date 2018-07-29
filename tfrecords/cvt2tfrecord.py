import os
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np

class time_images():
	def __init__(self, dirs):
		self.time_stamp = dirs.split('/')[-1]
		img_list = os.listdir(dirs)
		self.x_img = np.array(Image.open(os.path.join(dirs, img_list[0])))
		self.x_img = self.x_img.reshape((1, 501, 501, 1))
		for i in img_list[1:31]:
			full_i = os.path.join(dirs, i)
			print(i)
			img_i = np.array(Image.open(full_i)).reshape((1, 501, 501, 1))
			self.x_img = np.concatenate([self.x_img, img_i], axis=0)
			height_i, width_i = img_i.shape[0], img_i.shape[1]
		assert self.x_img.shape == (31, 501, 501, 1)
		self.height = height_i
		self.width = width_i
		self.x_img = self.x_img.tostring()

		self.y_img = np.array(Image.open(os.path.join(dirs, img_list[31])))
		self.y_img = self.y_img.reshape((1, 501, 501, 1))
		for i in img_list[32:]:
			full_i = os.path.join(dirs, i)
			print(i)
			img_i = np.array(Image.open(full_i)).reshape((1, 501, 501, 1))
			self.y_img = np.concatenate([self.y_img, img_i], axis=0)
			height_i, width_i = img_i.shape[0], img_i.shape[1]
		assert self.y_img.shape == (30, 501, 501, 1)
		self.y_img = self.y_img.tostring()

class cvter():
	def __init__(self, file_name, data_dir):
		self.writer = tf.python_io.TFRecordWriter(file_name)
		self.data_dirs = []
		l = os.listdir(data_dir)
		l = l[0:len(l)//25]
		print('before: ', len(l) * 25)
		print('after: ', len(l))
		for i in l:
			if os.path.isdir(os.path.join(data_dir, i)): 
				self.data_dirs.append(os.path.join(data_dir, i))
		self.time_imgs = []
		for i in self.data_dirs:
			print(i)
			self.time_imgs.append(self._read_imgs(i))

	def convert(self):
		idx = 1
		for imgs in self.time_imgs:
			example = tf.train.Example(features=tf.train.Features(feature={
				'data_raw': self._bytes_feature(imgs.x_img),
				'label_raw' : self._bytes_feature(imgs.y_img),
				'time_stamp': self._bytes_feature(imgs.time_stamp.encode('utf-8')),
				'height': self._int64_feature(imgs.height),
				'width' : self._int64_feature(imgs.width)
			}))
			print('Writing {}'.format(idx))
			self.writer.write(example.SerializeToString())
			idx += 1
		self.writer.close()

	def _read_imgs(self, dir_):
		return time_images(dir_)

	def _bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--records', type=str, default='01', help='the name of the output tfrecords name')
	parser.add_argument('--data_dir', type=str, default='../data', help='the dir to the data')
	args = parser.parse_args()
	cvt = cvter(file_name=args.records + '.tfrecords', data_dir=args.data_dir)
	cvt.convert()
