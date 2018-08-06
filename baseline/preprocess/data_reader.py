# this file contains script that build the data reading precedure into the graph
# for this project(TJCS Dev Group- pred-rain) the data has two parts, use the 
# previous 31 frames of radar map to produce prediction of the next 30 frame of radar map
# in the ICDM2018 competition only 6 frame is required to be evaluate (5, 10, 15, 20, 25, 30)

import os
import numpy as np
import random
from PIL import Image

class np_data_reader():
	def __init__(self, args, train_val_split=0.2, shuffle=True):
		self.data_dir = args.data_dir
		self.buffer_size = args.batch_size
		self.shuffle = shuffle
		self.all_img_dir = []
		l = os.listdir(self.data_dir)
		for i in l:
			if os.path.isdir(os.path.join(self.data_dir, i)):
				self.all_img_dir.append(os.path.join(self.data_dir, i))
		if self.shuffle is True:
			random.shuffle(self.all_img_dir)

		self.val_img_dirs = self.all_img_dir[0:int(train_val_split*len(self.all_img_dir))]
		self.all_img_dir = self.all_img_dir[int(train_val_split*len(self.all_img_dir)):]
		
		self.train_img_dir = self.all_img_dir[0:self.buffer_size]
		self.train_buffer_ptr = [0, self.buffer_size]
		self.train_buffer_time = 1
		self.train_n_steps = len(self.all_img_dir) // self.buffer_size
		
		self.val_img_dir = self.val_img_dirs[0:self.buffer_size]
		self.val_buffer_ptr = [0, self.buffer_size]
		self.val_buffer_time = 1
		self.val_n_steps = len(self.val_img_dirs) // self.buffer_size

	def read_train_data(self):
		x_img, y_img = self._read_img_from_dirs(self.train_img_dir[0])
		x_img = x_img.reshape(tuple([1] + list(x_img.shape)))
		y_img = y_img.reshape(tuple([1] + list(y_img.shape)))
		for i in self.train_img_dir[1:]:
			x_i, y_i = self._read_img_from_dirs(i)
			x_i = x_i.reshape(tuple([1] + list(x_i.shape)))
			y_i = y_i.reshape(tuple([1] + list(y_i.shape)))
			x_img = np.concatenate((x_img, x_i), axis=0)
			y_img = np.concatenate((y_img, y_i), axis=0)

		self._flush_train_buffer()
		return x_img, y_img

	def _flush_train_buffer(self):
		self.train_buffer_ptr[0] = self.train_buffer_ptr[1]
		self.train_buffer_ptr[1] += self.buffer_size
		self.train_buffer_time += 1
		if self.train_buffer_time >= self.train_n_steps:
			self.train_buffer_ptr = [0, self.buffer_size]
			self.train_buffer_time = 1
			if self.shuffle is True:
				random.shuffle(self.all_img_dir)
		self.train_img_dir = self.all_img_dir[self.train_buffer_ptr[0]:self.train_buffer_ptr[1]]

	def read_val_data(self):
		x_img, y_img = self._read_img_from_dirs(self.val_img_dir[0])
		x_img = x_img.reshape(tuple([1] + list(x_img.shape)))
		y_img = y_img.reshape(tuple([1] + list(y_img.shape)))
		for i in self.val_img_dir[1:]:
			x_i, y_i = self._read_img_from_dirs(i)
			x_i = x_i.reshape(tuple([1] + list(x_i.shape)))
			y_i = y_i.reshape(tuple([1] + list(y_i.shape)))
			x_img = np.concatenate((x_img, x_i), axis=0)
			y_img = np.concatenate((y_img, y_i), axis=0)

		self._flush_val_buffer()
		return x_img, y_img

	def _flush_val_buffer(self):
		self.val_buffer_ptr[0] = self.val_buffer_ptr[1]
		self.val_buffer_ptr[1] += self.buffer_size
		self.val_buffer_time += 1
		if self.val_buffer_time >= self.val_n_steps:
			self.val_buffer_ptr = [0, self.buffer_size]
			self.val_buffer_time = 1
			if self.shuffle is True:
				random.shuffle(self.val_img_dirs)
		self.val_img_dir = self.val_img_dirs[self.val_buffer_ptr[0]:self.val_buffer_ptr[1]]


	def _read_img_from_dirs(self, dirs):
		img_list = os.listdir(dirs)
		x_img = np.array(Image.open(os.path.join(dirs, img_list[0])))
		x_img = x_img.reshape((1, 501, 501, 1))
		for i in img_list[1:31]:
			full_i = os.path.join(dirs, i)
			img_i = np.array(Image.open(full_i)).reshape((1, 501, 501, 1))
			x_img = np.concatenate([x_img, img_i], axis=0)

		y_img = np.array(Image.open(os.path.join(dirs, img_list[31])))
		y_img = y_img.reshape((1, 501, 501, 1))
		for i in img_list[32:]:
			full_i = os.path.join(dirs, i)
			img_i = np.array(Image.open(full_i)).reshape((1, 501, 501, 1))
			y_img = np.concatenate([y_img, img_i], axis=0)
		assert x_img.shape == (31, 501, 501, 1)
		assert y_img.shape == (30, 501, 501, 1)
		return x_img, y_img

if __name__ == '__main__':
	pass