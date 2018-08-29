from utils.imports import *
from input_fn import input_fn
from cvt2tfrecord import fn_record_to_count
from utils.clr import LRFinder, OneCycleLR
from tensorflow.keras.optimizers import SGD

MODEL_NAME = 'prednet'

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
        # from utils.transforms import *
        aug_tfms = []
        return aug_tfms 
    
    
    def generate_stats(self, par_dirs, PATH=None):
        if PATH is None:        # if called with an instance
            PATH = self.PATH
            
        from data_reader import calc_mean_std_par_dir, write_mean_std
        # TODO: parallelize?
        stats = [calc_mean_std_par_dir(par_dir) for par_dir in par_dirs]
        means, stds = zip(*stats)
        
        dir_stats = self.PATH/'stats'
        dir_stats.mkdir(exist_ok=True)
        
        name_stats = [fn.name for fn in par_dirs]
        fn_stats = [dir_stats/name_stat for name_stat in name_stats]
        for stat in zip(fn_stats, means, stds):
            write_mean_std(*stat)
        
        
    def get_data(self, base_path=Path('../data/tfrecords'), mode='contiguous', idx_list=[1],
                 val_split=0.1, num_parallel_calls=32, buffer_size=3):
        # train_1_contiguous_10
        fn_records = ['train_' + str(idx) + '_' + str(mode) + '_' + str(self.nt) for idx in idx_list]
        fns = [str(Path(base_path)/o) for o in fn_records]
        
        # TODO: generate fn_stats
        fn_stats = PATH/'stat.csv'
        
        trn_tensors = input_fn(bs, sz, nt, aug_tfms, fns, is_val=False, val_split=val_split,
                                     stats_fn='stat.csv', num_parallel_calls=num_parallel_calls, shuffle=True,
                                     buffer_size=buffer_size)
        val_tensors = input_fn(bs, sz, nt, aug_tfms, fns, is_val=True, val_split=val_split,
                                     stats_fn='stat.csv', num_parallel_calls=num_parallel_calls, shuffle=False,
                                     buffer_size=buffer_size)
        
        return trn_tensors, val_tensors
        

    def get_model(self, optimizer=SGD(lr=0.002, momentum=0.9, nesterov=True)):
        # TODO: add experiments to argument list
        
        from models.prednet_refactored import PredNetCell, PredNet
        n_channels, im_height, im_width = (1, sz, sz)
        input_shape = (im_height, im_width, n_channels)
        stack_sizes = (n_channels, 48, 96, 192)
        R_stack_sizes = stack_sizes
        A_filt_sizes = (3, 3, 3)
        Ahat_filt_sizes = (3, 3, 3, 3)
        R_filt_sizes = (3, 3, 3, 3)

        layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; 
                                                        # "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
        layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
        time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
        time_loss_weights[0] = 0

        prednet_cell = PredNetCell(stack_sizes=stack_sizes,
                    R_stack_sizes=R_stack_sizes,
                    A_filt_sizes=A_filt_sizes,
                    Ahat_filt_sizes=Ahat_filt_sizes,
                    R_filt_sizes=R_filt_sizes)
        prednet = PredNet(prednet_cell)
        
        from tensorflow.keras.layers import TimeDistributed, Flatten, Dense
        from tensorflow.keras.models import Model

        inputs = tf.keras.Input(shape=(nt,) + input_shape)
        errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers) 
        errors_by_time = TimeDistributed(Dense(1, trainable=False), 
                                         weights=[layer_loss_weights, np.zeros(1)], 
                                         trainable=False)(errors)  # calculate weighted error by layer
        errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
        final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
        model = Model(inputs=inputs, outputs=final_errors)
        if self.num_gpus > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=self.num_gpus)
        model.compile(loss='mean_absolute_error', optimizer=optimizer) 
        
        return model
    
    

if __name__ == '__main__':
    fire.Fire(Train)