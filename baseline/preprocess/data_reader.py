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
            param:
                sess: the current tf.Session()
                batch_size: the batch_size
                prepro_func: a data preprocessing(augmentation) function
                            takes two param (img, label), 
                            img is a (None, 31, 501, 501, 1) tensor,
                            label is a (None, 30, 501, 501, 1) tensor
                            returns the precessed img and label
                            e.g
                                def identity(img, label):
                                    return img, label  
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
            param: 
                mode: choose between 'train' and 'valid'
            return a feed_dict needed to feed into the sess
        '''
        # TODO: test phase
        if mode == 'train':
            return {self.handle: self.train_handle}
        elif mode == 'valid':
            return {self.handle: self.vali_handle}
        else:
            raise NotImplementedError('Wrong Mode!!!')

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
