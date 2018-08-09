# this file contains the main script to train the model and produce prediction from the 
# ICDM2018 competition test data

import argparse
import tensorflow as tf
import numpy as np
from models.baseline_model import ConvLSTM_Model
from preprocess.data_reader import data_reader

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--train_val_split', type=float, default=0.2)
    parser.add_argument('--train_records', type=str, default='train_01.tfrecords')
    parser.add_argument('--valid_records', type=str, default='vali_01.tfrecords')
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
    dr = data_reader(args)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = ConvLSTM_Model(args)
    X, y = dr.read_and_decode(sess, args.batch_size)
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


