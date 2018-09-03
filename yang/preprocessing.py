from utils.imports import *

def convert_to_tf_records(base_path, idx, PATH, is_test, pred_mode, nt=None, stop=None):
    """
    Arguments: 
        base_path: the path to the parent folder of datasets. e.g. '../data/SRAD2018'
        idx: the idx of datasets
        PATH: working directory
        is_test
        pred_mode: either contiguous or skip
        nt: number of time steps
    """
    print(pred_mode)
    fn_dirs = [Path(base_path)/o for o in fn_idx_to_dir_names(is_test, idx)]
    fn_records = fn_to_record(Path(PATH)/'tfrecords', pred_mode, is_test, idx, nt)
    for fn_dir in fn_dirs:
        assert Path(fn_dir).exists(), 'data not found'

    if not is_test:
        if pred_mode == 'skip': 
            assert stop is not None
        elif pred_mode == 'contiguous':
            assert nt is not None
        else:
            raise ValueError('wrong pred_mode')

    for fn_dir, fn_record in tqdm_notebook(zip(fn_dirs, fn_records)):
        writer = tf.python_io.TFRecordWriter(fn_record)
        subdirs = sorted(Path(fn_dir).iterdir())
        num_samples = 0
        for subdir in tqdm_notebook(subdirs, desc=f'Processing subdirectory'):
            if subdir.is_dir(): # for subdirecories
                time_stamp = subdir.name
                fns = np.asarray(sorted(subdir.glob('*.png')))
                 
                if pred_mode == 'contiguous':
                    fns_split = [fns[o:o+nt] for o in range(0, len(fns) - nt + 1, nt)]
                elif pred_mode == 'skip':
                    if is_test:
                        # hard coding conversion method
                        fns_split = [fns[range(o, 31, 5)] for o in range(1, 6)]
                    else:
                        split_idx = [range(o, len(fns), stop) for o in range(stop)]
                        if nt is not None:
                            split_idx = [i[o:o + nt] 
                                         for i in split_idx
                                         for o in range(0, len(i) - nt + 1, nt)]
                        fns_split = [fns[idx] for idx in split_idx]
                else:
                    raise NotImplementedError

                for fn_sequence in fns_split: # read images
                    raw_png = []
                    for fn in fn_sequence:
                        with open(str(fn), 'rb') as raw_png_open:
                            raw_png.append(raw_png_open.read())

                    # execute for each sequence
                    context = tf.train.Features(feature={
                        'time_stamp': _bytes_feature(time_stamp.encode())
                    })
                    feature_lists = tf.train.FeatureLists(
                        feature_list={
                            'raw_png': _bytes_feature_list(raw_png)
                        }
                    )
                    sequence_example = tf.train.SequenceExample(context=context,
                                                            feature_lists=feature_lists)
                    writer.write(sequence_example.SerializeToString())
                    
            
def generate_stats(par_par_dir, is_test, idx, counts):
    """ Generate the number of sequences, mean and std for each dataset and write to disk

    Arguments:
        par_par_dir: while par_dir stands for the path to a dataset, the par_par_dir is 
            the path to the parent directory of the datasets
            e.g. ../data/SRAD2018/
        is_test
        idx: a comma seperated list of datasets
        PATH: working directory that will contain the stats folder (e.g. ../data/)
    """
    par_dirs = [Path(par_par_dir)/o for o in fn_idx_to_dir_names(is_test, idx)]

    stats = [calc_mean_std_par_dir(str(par_dir)) 
             for par_dir in tqdm_notebook(par_dirs)]
    means, stds = zip(*stats)

    counts = [len(list(o.glob('RAD_*'))) for o in par_dirs]

    fn_stats = fn_idx_to_stats(PATH, is_test, idx)

    dir_stats = Path(PATH)/'stats'
    if not dir_stats.exists():
        dir_stats.mkdir()
    for stat in zip(fn_stats, means, stds, counts):
        write_mean_std_count(*stat)
                  
                
# Helper functions
def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto,
e.g, An integer label.
"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto,
e.g, an image in byte
"""
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
e.g, sentence in list of ints
"""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
e.g, sentence in list of bytes
"""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


if __name__ == '__main__':
    fire.Fire(convert_to_tf_records)