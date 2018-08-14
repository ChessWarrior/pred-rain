# this file contains script that convert the data file into a single tfrecords file (easier to be build into graph)

import tensorflow as tf
from utils.imports import *
import argparse

class Cvter():
    def __init__(self, file_name, data_dir):
        self.file_name, self.data_dir = file_name, data_dir

    def convert_continious(self):
        writer = tf.python_io.TFRecordWriter(self.file_name)
        subdirs = sorted(list(Path(self.data_dir).iterdir()))
        for subdir in tqdm_notebook(subdirs, desc='Processing subdirectory'):
            if subdir.is_dir(): # for subdirecories
                time_stamp = subdir.name
                raw_png = []
                fns = sorted(subdir.glob('*.png'))
                for i, fn in enumerate(fns): # read images
                    with open(str(fn), 'rb') as raw_png_open:
                        raw_png.append(raw_png_open.read())
                
                # execute for each subdirectory
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
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    file_name = args.recordskj + '.tfrecords'
    cvter = Cvter(file_name, args.data_dir)
    cvter.convert_continious()