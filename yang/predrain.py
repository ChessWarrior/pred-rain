from utils.imports import *
from utils.clr import LRFinder, OneCycleLR
from tensorflow.keras.optimizers import SGD
from utils.transforms import *

from models.model_factory import model_factory, ModelType

# TODO: pylint it

class Predrain():
    
    def __init__(self):
        self.initialized = False
    
    
    def set_flags(self, sz, nt, bs, version, num_gpus, gpu_start):
        self.num_gpus = num_gpus
        self.sz, self.nt, self.bs = sz, nt, bs
        
        gpu_range = range(gpu_start, gpu_start + num_gpus)
        gpu_range_str = '0' if num_gpus == 1 else ','.join(gpu_range)
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'     # so the IDs match nvidia-smi
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_range_str # "0, 1" for multiple

        self.PATH = Path('../data')
        self.MODEL_VERSION = MODEL_NAME + '_' + str(sz) + '_' + str(version)
        self.MODEL_PATH = PATH/'models'
        
        self.path_checkpoints = self.MODEL_PATH/'checkpoints'
        self.path_lrs = self.MODEL_PATH/'lrs'/self.MODEL_VERSION
        if not path_checkpoints.exists(): path_checkpoints.mkdir(parents=True)
        
        self.initialized = True
        
    
    def train(self, sz=128, nt=10, bs=12, version=1, num_gpus=1, gpu_start=0):
        self.set_flags(sz=sz, nt=nt, bs=bs, version=version, num_gpus=num_gpus, gpu_start=gpu_start)
        
        aug_tfms = self.get_tfms()
        
        model = self.get_model()
        
        trn_tensors, val_tensors = self.get_data()
        
        self.fit()
        
        
    def fit(self, trn_tensors, val_tensors, epochs=1, max_lr=1e-3 / 2):
        """ 
        Aruguments:
            trn_tensors and val_tensors are return by self.get_data
        """
        model_callbacks = [
            # TODO: save weight info of prednet cell
            tf.keras.callbacks.TensorBoard(),
        #     keras.callbacks.History(),
        #     tf.keras.callbacks.ModelCheckpoint(str(path_checkpoints/('weights.' + MODEL_VERSION + '.{epoch:02d}-{val_loss:.2f}.hdf5'))),
            tf.keras.callbacks.ModelCheckpoint(str(path_checkpoints/('weights.' + MODEL_VERSION + '.{epoch:02d}.hdf5'))),
        #     keras.callbacks.EarlyStopping()
        ]
        
        from utils.clr import OneCycleLR
        lr_manager = OneCycleLR(num_samples, 1, self.bs, max_lr,
                                end_percentage=0.1, scale_percentage=None,
                                maximum_momentum=0.95, minimum_momentum=0.85, verbose=True)
        
        (x, y, num_samples), (val_x, val_y, val_num_samples) = trn_tensors, val_tensors
        trn_steps, val_steps = [int(epochs * o / self.bs) for o in (num_samples, val_num_samples)]
        model.fit(x, y, epochs=1, validation_data=val_tensor, callbacks=callbacks, 
                  steps_per_epoch=trn_steps, validation_steps=val_steps)
        
        
    def find_lr(self, x, y, num_samples, lr_pct=0.01):
        lr_num_samples = num_samples * lr_pct
        epochs = 1
        lr_steps = int(epochs * lr_num_samples / self.bs)

        from utils.clr import LRFinder
        lrfinder = LRFinder(lr_num_samples, self.bs, save_dir=str(self.path_checkpoints))
        lr_callbacks = [lrfinder]
        model.fit(x, y, steps_per_epoch=lr_steps, callbacks=lr_callbacks)
        lrfinder.plot_schedule()
        
        
    def get_tfms(self):
        # data augmentation
        # not implemented yet. see the TODO section in README.md
        # fro
        fn_stats = fn_idx_to_stats(PATH)
        stats = np.fromfile(stats_fn, sep=stats_sep) # normalization stat
        tfms, _ = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, crop_type=CropType.NO)
        aug_tfms = []
        return aug_tfms 
    
        
    def generate_stats(self, par_par_dir, is_test, idx, PATH=None):
        """ Generate the number of sequences, mean and std for each dataset and write to disk
        
        Arguments:
            par_par_dir: while par_dir stands for the path to a dataset, the par_par_dir is 
                the path to the parent directory of the datasets
                e.g. ../data/SRAD2018/
            is_test
            idx: a comma seperated list of datasets
            PATH: working directory that will contain the stats folder (e.g. ../data/)
        
        Returns:
            The stats
        """
        if PATH is None:        # if called with an instance
            PATH = self.PATH
            
        # parallelize?
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
            
        
    import multiprocessing
    def get_data(self, base_path=Path('../data/tfrecords'), pred_mode='contiguous', 
                 idx_list=[1], val_split=0.1, num_parallel_calls=multiprocessing.cpu_count(),
                 buffer_size=1, shuffle_buffer_size=24, is_test=False):
        fns_records = fn_to_record(base_path, pred_mode, is_test, idx_list, self.nt)
        
        stats = get_stats(self.PATH, idx_list, is_test)
        if len(idx_list) == 1:
            mean, std, num_samples = stats[0]
        else:
            raise NotImplementedError('averate stats')
            
        if not is_test:
            trn_tensors = input_fn(fns, is_val=False, shuffle=True, val_split=val_split,
                                   stats=stats, num_parallel_calls=num_parallel_calls,
                                   buffer_size=buffer_size,
                                   shuffle_buffer_size=shuffle_buffer_size)
        else:
            trn_tensors = None
        
        val_tensors = input_fn(fns, is_val=True, shuffle=False, val_split=val_split,
                               stats=stats, num_parallel_calls=num_parallel_calls, 
                               buffer_size=buffer_size,
                               shuffle_buffer_size=shuffle_buffer_size)
        
        return trn_tensors, val_tensors
        

    def get_model(self, model_type, **args):
        # TODO: 
        #   1. add experiments to argument list
        #   2. load models from files
        
        # This method helps filling up basic model information
        args['num_spus'] = self.num_gpus
        args['sz'] = self.sz
        args['nt'] = self.nt
        args['bs'] = self.bs
        args['MODEL_VERSION'] = self.MODEL_VERSION
        args['MODEL_PATH'] = self.MODEL_PATH
        args['path_checkpoints'] = self.path_checkpoints
        
        model = model_factory(model_type, **args)
        return model
    
    
    def input_fn(self, fns, is_val, shuffle, val_split, stats, num_parallel_calls, 
                 buffer_size, shuffle_buffer_size):
        """
        TODO: data interleave
        Create tf.data.Iterator from tfrecord file.
        """
        fn_records = [o + '.tfrecords' for o in fns]
        dataset = tf.data.TFRecordDataset(fn_records)

        def parser_train(serialized_example):
            # experimental. TODO: read only needed samples
            shape = (nt, 501, 501, 3)
            context_features = {
                    'time_stamp': tf.FixedLenFeature([], tf.string),
                }
            sequence_features = {
                    "raw_png": tf.FixedLenSequenceFeature([], dtype=tf.string)
                }

            features, sequence_features = tf.parse_single_sequence_example(
                serialized_example, 
                context_features=context_features, 
                sequence_features=sequence_features)

            x = tf.map_fn(tf.image.decode_png, sequence_features['raw_png'], dtype=tf.uint8,
                        back_prop=False, swap_memory=False, infer_shape=False)
            x = tf.cast(x, tf.float32)
            x /= 255
            x.set_shape(shape)
            x = tf.expand_dims(x[:,:,:,0], axis=3)
            return x 

        # Train val split
        num_shards = 1 / val_split
        val_idx = num_shards - 1
        shard_idx = int(val_idx if is_val else 0)
        dataset = dataset.shard(int(num_shards), shard_idx)

        if shuffle:
            dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size))
        else:
            dataset.repeat()
        dataset = dataset.map(parser_train, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(tfms, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(bs)
        dataset = dataset.prefetch(buffer_size)

        y = tf.zeros([bs, 1])
        iterator = dataset.make_one_shot_iterator()
        x = iterator.get_next()

        fn_counts = [fn_record_to_count(o) for o in fns]
        num_samples = 0
        for fn_count in fn_counts:
            with open(fn_count, 'r') as f:
                num_samples += int(f.read())
        num_pct = val_split if is_val else (1 - val_split)
        num_samples *= num_pct

        return x, y, int(num_samples)
    
    
    @classmethod
    def convert_to_tf_records(cls, base_path, idx, PATH, is_test, pred_mode, nt=None,
                              stop=None):
        """
        Arguments: 
            base_path: the path to the parent folder of datasets. e.g. '../data/SRAD2018'
            idx: the idx of datasets
            PATH: working directory
            is_test
            pred_mode: either contiguous or skip
            nt: number of time steps
        """
        fn_dirs = [Path(base_path)/o for o in fn_idx_to_dir_names(is_test, idx)]
        fn_records = fn_to_record(Path(PATH)/'tfrecords', pred_mode, is_test, idx, nt)
        for fn_dir in fn_dirs:
            assert Path(fn_dir).exists(), 'data not found'
            
        if pred_mode == 'skip': 
            assert stop is not None
        else:
            assert nt is not None
            
        for fn_dir, fn_record in tqdm_notebook(zip(fn_dirs, fn_records)):
            writer = tf.python_io.TFRecordWriter(fn_record)
            subdirs = sorted(Path(fn_dir).iterdir())
            for subdir in tqdm_notebook(subdirs, desc=f'Processing subdirectory'):
                if subdir.is_dir(): # for subdirecories
                    time_stamp = subdir.name
                    fns = np.asarray(sorted(subdir.glob('*.png')))

                    if pred_mode == 'contiguous':
                        fns_split = [fns[o:o+nt] for o in range(0, len(fns) - nt, nt)]
                    elif pred_mode == 'skip':
                    # Caveat: think about how to recover the order of prediction when 
                    # the result is truncated.
                        split_idx = [range(o, len(fns) - stop, stop) for o in range(stop)]
                        if nt is not None:
                            split_idx = [i[o:o + nt] 
                                         for i in split_idx 
                                         for o in range(0, len(i) - nt, nt)]
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
    fire.Fire(Predrain)