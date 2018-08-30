from .prednet_refactored import *
from utils.imports import *
from enum import IntEnum

class ModelType(IntEnum):
    PredNetOriginal = 1
    PredNetV1 = 2

def model_factory(model_type, **args):
    if model_type == ModelType.PredNetOriginal:
        # Requires: sz, output_mode, num_gpus
        n_channels, im_height, im_width = (1, args.sz, args.sz)
        input_shape = (im_height, im_width, n_channels)
        stack_sizes = (n_channels, 48, 96, 192)
        R_stack_sizes = stack_sizes
        A_filt_sizes = (3, 3, 3)
        Ahat_filt_sizes = (3, 3, 3, 3)
        R_filt_sizes = (3, 3, 3, 3) 
        prednet_cell = PredNetCell(stack_sizes=stack_sizes,
                                  R_stack_sizes=R_stack_sizes,
                                  A_filt_sizes=A_filt_sizes,
                                  Ahat_filt_sizes=Ahat_filt_sizes,
                                  R_filt_sizes=R_filt_sizes,
                                  output_mode=args.output_mode)
        prednet = PredNet(prednet_cell)
        
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
        if args.num_gpus > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=args.num_gpus)
        optimizer=SGD(lr=0.002, momentum=0.9, nesterov=True)
        model.compile(loss='mean_absolute_error', optimizer=optimizer) 
        
        return model
    
    elif model_type == ModelType.PredNetV1:
        raise NotImplementedError
        
    else:
        raise NotImplementedError