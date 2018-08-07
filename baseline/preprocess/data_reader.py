# This file contains scripts that build the data reading precedure into the graph
# for this project (TJCS Dev Group- pred-rain). The data has two parts, using the 
# first 31 frames of radar images to produce predictions of the next 30 frames of radar images
# In the ICDM2018 competition only 6 frames are required for evaluation (5, 10, 15, 20, 25, 30).

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

class data_reader():
    def __init__(self, args):
        self.args = args
        self.train_file_name = args.train_records
        self.vali_file_name = args.valid_records
        self.train_dataset = tf.data.TFRecordDataset(self.train_file_name)
        self.vali_dataset = tf.data.TFRecordDataset(self.vali_file_name)
        self.handle = tf.placeholder(shape=[], dtype=tf.string)

        # TODO: Now the steps are for only one zip file(contains 5000 dirs)
        # Change the numbers acording to needs
        self.val_n_steps = int((5000 * self.args.train_val_split) // self.args.batch_size)
        self.train_n_steps = int((5000 // self.args.batch_size) - self.val_n_steps)

    def read_and_decode(self, sess, batch_size, prepro_func=None):
        '''
            return the tensors needed to build the model
        '''
        def identity(img, label):
            return img, label

        def parser(record):
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

        if prepro_func == None:
            prepro_func = identity

        self.train_dataset = self.train_dataset.map(parser).map(prepro_func).repeat().batch(batch_size)
        self.vali_dataset = self.vali_dataset.map(parser).map(prepro_func).repeat().batch(batch_size)

        self.iterator = tf.data.Iterator.from_string_handle(
                                        self.handle, 
                                        self.train_dataset.output_types, 
                                        self.train_dataset.output_shapes)

        self.train_iterator = self.train_dataset.make_one_shot_iterator()
        self.vali_iterator = self.vali_dataset.make_one_shot_iterator()
        self.train_handle = sess.run(self.train_iterator.string_handle())
        self.vali_handle = sess.run(self.vali_iterator.string_handle())

        img_input, label = self.iterator.get_next()
        return img_input, label

    def set_mode(self, mode='train'):
        '''
            return a feed_dict needed to feed into the sess
        '''
        if mode == 'train':
            return {self.handle: self.train_handle}
        elif mode == 'valid':
            return {self.handle: self.vali_handle}
        else:
            raise NotImplementedError('Wrong Mode!!!')

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

# Yang
# Funcitons to support reading from sequence tfrecord file
# Usage:
    # dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.map(parser_train)
    # iterator = dataset.make_initializable_iterator()
    # sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    # x, y, time_stamp = iterator.get_next()
def parser_train(serialized_example):
    features = {
        'data_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
    }
    features, sequence_features = tf.parse_single_sequence_example(
        serialized_example, context_features={
            'time_stamp': tf.FixedLenFeature([], tf.string),
        }, sequence_features={
            "data_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "label_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
        })
    cast_to_float32 = partial(tf.cast, dtype=tf.float32)
    
    x = tf.map_fn(tf.image.decode_png, sequence_features['data_raw'], dtype=tf.uint8,
                back_prop=False, swap_memory=False, infer_shape=False)
    x = tf.map_fn(cast_to_float32, x, dtype=tf.float32,
                back_prop=False, swap_memory=False, infer_shape=False)
    x = tf.squeeze(x)
    
    y = tf.map_fn(tf.image.decode_png, sequence_features['label_raw'], dtype=tf.uint8,
                back_prop=False, swap_memory=False, infer_shape=False)
    y = tf.map_fn(cast_to_float32, y, dtype=tf.float32,
                back_prop=False, swap_memory=False, infer_shape=False)
    y = tf.squeeze(y)
    
    return x, y, features['time_stamp']


if __name__ == '__main__':
    pass
