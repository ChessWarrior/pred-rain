from utils.imports import *
from utils.clr import LRFinder, OneCycleLR
from tensorflow.keras.optimizers import SGD
from utils.transforms import *

from models.model_factory import model_factory, ModelType

# TODO: pylint it

class Predrain():
    
    def __init__(self):
        self.initialized = False
    
    def set_config(self, sz, nt, bs, model_type, num_gpus, gpu_start, pred_mode, comment='', PATH=None, allow_growth=False):
        if isinstance(model_type, int):
            model_type = ModelType(model_type)
        PATH = PATH or '../data'
        self.PATH = Path(PATH)
        self.num_gpus, self.gpu_start = num_gpus, gpu_start
        self.sz, self.nt, self.bs = sz, nt, bs
        
        gpu_range = range(gpu_start, gpu_start + num_gpus)
        gpu_range_str = str(gpu_start) if num_gpus == 1 else ','.join([str(o) for o in gpu_range])
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'     # so the IDs match nvidia-smi
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_range_str # "0, 1" for multiple

        self.MODEL_VERSION = str_version(model_type, pred_mode, sz, comment)
        self.MODEL_PATH = self.PATH/'models'
        
        self.path_checkpoints = self.MODEL_PATH/'checkpoints'
        self.path_lrs = self.MODEL_PATH/'lrs'/self.MODEL_VERSION
        self.path_logs = self.MODEL_PATH/'logs'/self.MODEL_VERSION
        for p in [self.path_checkpoints, self.path_lrs, self.path_logs]:
               if not p.exists(): p.mkdir(parents=True) 
        
        if allow_growth:
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                                # (nothing gets printed in Jupyter, only if you run it standalone)
            sess = tf.Session(config=config)
            set_session(sess)  # set this TensorFlow session as the default session for Keras
        
        #self.initialized = True
        self.has_data, self.has_model = False, False
        self.model = None
        self.trn_tensors, self.val_tensors, self.num_samples, self.val_num_samples = None, None, None, None
        
        self.initialized = True
        
    
    def train(self, epochs=1, max_lr=1e-3 / 2):
        if not self.has_model:
            model = self.get_model(ModelType.PredNetOriginal)
        if not self.has_data:
            trn_tensors, val_tensors, num_samples, val_num_samples = self.get_data()
            
        # clear tensorboard folder
        shutil.rmtree(str(self.path_logs))
        self.path_logs.mkdir()
        
        self.fit(epochs, max_lr)
        print(self.MODEL_VERSION)
        
        
    def fit(self, epochs, max_lr, **args):
        """ 
        Aruguments:
            trn_tensors and val_tensors are tensor tuples of (x, y)
        """
        assert self.has_model and self.has_data
        model_callbacks = [
            # TODO: save weight info of prednet cell
            tf.keras.callbacks.TensorBoard(log_dir=str(self.path_logs), histogram_freq=1, batch_size=self.bs,
                                          write_images=True),
            tf.keras.callbacks.ModelCheckpoint(str(self.path_checkpoints/('weights.' + self.MODEL_VERSION + '.{epoch:02d}.h5')), save_best_only=True, save_weights_only=True),
        ]
        
        lr_manager = OneCycleLR(self.num_samples, epochs, self.bs, max_lr,
                                end_percentage=0.1, scale_percentage=None,
                                maximum_momentum=0.95, minimum_momentum=0.85, verbose=True)
        callbacks = model_callbacks + [lr_manager]
        #callbacks = model_callbacks
        
        trn_steps, val_steps = [int(epochs * o / self.bs) 
                                for o in (self.num_samples, self.val_num_samples)]
        x, y = self.trn_tensors
        self.model.fit(x, y, epochs=epochs, validation_data=self.val_tensors, callbacks=callbacks, 
                  steps_per_epoch=trn_steps, validation_steps=val_steps, **args)
    
    def predict(self):
        assert self.has_model
        
        
    def lr_find(self, lr_pct=0.05):
        assert self.has_model and self.has_data
        lr_num_samples = self.num_samples * lr_pct
        lr_steps = int(lr_num_samples / self.bs)

        lrfinder = LRFinder(lr_num_samples, self.bs, save_dir=str(self.path_lrs))
        lr_callbacks = [lrfinder]
        x, y = self.trn_tensors
        
        fn_tmp = str(self.path_lrs/'tmp.h5')
        self.model.save_weights(fn_tmp)
        self.model.fit(x, y, steps_per_epoch=lr_steps, callbacks=lr_callbacks)
        self.model.load_weights(fn_tmp)
        
        return lrfinder
        
        
    import multiprocessing
    def get_data(self, base_path=Path('../data/tfrecords'), pred_mode='contiguous', 
                 idx=[1], val_split=0.1, num_parallel_calls=multiprocessing.cpu_count(),
                 buffer_size=3, shuffle_buffer_size=24, is_test=False, aug_tfms=[]):
        if isinstance(idx, int):
            idx = [idx]
        fns_records = fn_to_record(base_path, pred_mode, is_test, idx, self.nt)
        
        num_samples = get_num_samples(len(idx), pred_mode, is_test)
        self.val_num_samples = int(num_samples * val_split)
        self.num_samples = int(num_samples - self.val_num_samples)
            
        mean, std = 0.22, 0.90
        # provides normalization and resizing (and optionally croping)
        tfms, _ = tfms_from_stats((mean, std), self.sz, aug_tfms=aug_tfms, 
                                  crop_type=CropType.NO)
        self.tfms = tfms
        self.denorm = Denormalize(mean, std)
       
        if not is_test:
            self.trn_tensors = self._input_fn(fns_records, is_val=False, shuffle=True,
                                            val_split=val_split, tfms=tfms,
                                            num_parallel_calls=num_parallel_calls,
                                            buffer_size=buffer_size,
                                            shuffle_buffer_size=shuffle_buffer_size)
        else:
            self.trn_tensors = None
        
        if is_test:
            val_split = 1
        self.val_tensors = self._input_fn(fns_records, is_val=True, shuffle=False,
                                        val_split=val_split, tfms=tfms,
                                        num_parallel_calls=num_parallel_calls, 
                                        buffer_size=buffer_size,
                                        shuffle_buffer_size=shuffle_buffer_size)
        self.has_data = True
        
        return self.trn_tensors, self.val_tensors, self.num_samples, self.val_num_samples
        

    def get_model(self, model_type, **args):
        # TODO: 
        #   1. add experiments to argument list
        #   2. load models from files
        
        # This method helps filling up basic model information
        args['num_gpus'] = self.num_gpus
        args['sz'] = self.sz
        args['nt'] = self.nt
        args['bs'] = self.bs
        args['MODEL_VERSION'] = self.MODEL_VERSION
        args['MODEL_PATH'] = self.MODEL_PATH
        args['path_checkpoints'] = self.path_checkpoints
       
        self.model = model_factory(model_type, **args)
        self.has_model = True
        return self.model
    
    
    def load(self, mt, old_sz, pred_mode, idx, comment='', **args):
        assert self.has_model
        weight_name = 'weights.' + str_version(mt, pred_mode, old_sz, comment) + '.' + str(idx).zfill(2) + '.h5'
        path_weight = str(self.path_checkpoints/weight_name)
        self.model.load_weights(path_weight, **args)
        
        
    def _input_fn(self, fns, is_val, shuffle, val_split, tfms, num_parallel_calls, 
                 buffer_size, shuffle_buffer_size):
        """
        TODO: data interleave
        Create tf.data.Iterator from tfrecord file.
        """
        dataset = tf.data.TFRecordDataset(fns)

        def parser_train(serialized_example):
            shape = (self.nt, 501, 501, 3)
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
        dataset = dataset.skip(self.num_samples) if is_val else dataset.take(self.num_samples)

        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size))
        else:
            dataset = dataset.repeat()
        dataset = dataset.map(parser_train, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(tfms, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(self.bs)
        dataset = dataset.prefetch(buffer_size)

        y = tf.zeros([self.bs, 1])
        iterator = dataset.make_one_shot_iterator()
        x = iterator.get_next()

        return x, y
    
    
if __name__ == '__main__':
    fire.Fire(Predrain)