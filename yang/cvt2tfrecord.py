# this file contains script that convert the data file into a single tfrecords file (easier to be build into graph)

import tensorflow as tf
from utils.imports import *
import argparse

class Cvter():
    def __init__(self, file_name, data_dir):
        self.file_name, self.data_dir = file_name, data_dir

    def convert_contiguous(self, nt):
        # write sequences of length nt to the tfrecords file
        # remainder frames are discarded
        writer = tf.python_io.TFRecordWriter(self.file_name)
        subdirs = sorted(list(Path(self.data_dir).iterdir()))
        for subdir in tqdm_notebook(subdirs, desc=f'Processing subdirectory {self.data_dir}'):
            if subdir.is_dir(): # for subdirecories
                time_stamp = subdir.name
                fns = sorted(subdir.glob('*.png'))
                
                fns_split = [fns[o:o+nt] for o in range(0, len(fns) - nt, nt)]
                for fn_sequence in fns_split: # read images
                    raw_png = []
                    for fn in fn_sequence:
                        with open(str(fn), 'rb') as raw_png_open:
                            raw_png.append(raw_png_open.read())
                
                    # execute for each sequence
                    context = tf.train.Features(feature={
                        'time_stamp': self._bytes_feature(time_stamp.encode())
                    })
                    feature_lists = tf.train.FeatureLists(
                        feature_list={
                            'raw_png': self._bytes_feature_list(raw_png)
                        }
                    )
                    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                    writer.write(sequence_example.SerializeToString())

    def convert_skipping(self, stop, nt=None):
        # write skipping sequences of __stop__ stop to the tfrecords file
        # if nt is not None, skipping sequences are split into subsequences of length nt
        # Caveat: think about how to recover the order of prediction when the result is truncated.
        writer = tf.python_io.TFRecordWriter(self.file_name)
        subdirs = sorted(list(Path(self.data_dir).iterdir()))
        for subdir in tqdm_notebook(subdirs, desc=f'Processing subdirectory {self.data_dir}'):
            if subdir.is_dir(): # for subdirecories
                time_stamp = subdir.name
                fns = np.asarray(sorted(subdir.glob('*.png')))
                
                split_idx = [range(o, len(fns) - stop, stop) for o in range(stop)]
                if nt is not None:
                    split_idx = [idx[o:o + nt] for idx in split_idx for o in range(0, len(idx) - nt, nt)]

                fns_split = [fns[idx] for idx in split_idx]
                for fn_sequence in fns_split: # read images
                    raw_png = []
                    for fn in fn_sequence:
                        with open(str(fn), 'rb') as raw_png_open:
                            raw_png.append(raw_png_open.read())
                
                    # execute for each sequence
                    context = tf.train.Features(feature={
                        'time_stamp': self._bytes_feature(time_stamp.encode())
                    })
                    feature_lists = tf.train.FeatureLists(
                        feature_list={
                            'raw_png': self._bytes_feature_list(raw_png)
                        }
                    )
                    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                    writer.write(sequence_example.SerializeToString())

    @classmethod    
    def _int64_feature(cls, value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto,
    e.g, An integer label.
    """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @classmethod    
    def _bytes_feature(cls, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto,
    e.g, an image in byte
    """
        # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @classmethod    
    def _int64_feature_list(cls, values):
        """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
    e.g, sentence in list of ints
    """
        return tf.train.FeatureList(feature=[cls._int64_feature(v) for v in values])

    @classmethod    
    def _bytes_feature_list(cls, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
    e.g, sentence in list of bytes
    """
        return tf.train.FeatureList(feature=[cls._bytes_feature(v) for v in values])



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--records', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--nt', type=int)
    parser.add_argument('-stop', type=str)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if not args.stop:
        args.stop = None
    else:
        args.stop = int(args.stop)

    file_name = args.records + '.tfrecords'
    cvter = Cvter(file_name, args.data_dir)
    if args.mode == 'contiguous':
        cvter.convert_contiguous(args.nt)
    elif args.mode == 'skip':
        cvter.convert_skipping(args.stop, args.nt)
    else:
        raise ValueError
    print(args.mode, args.stop, args.nt)
    print('Done!')
