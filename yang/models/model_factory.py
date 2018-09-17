from .prednet_refactored import *
from .dualprednet import *
from .darkprednet import *

from utils.imports import *
from enum import IntEnum
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# TODO: use dictionary to get model
class ModelType(IntEnum):
    PredNetOriginal = 1
    PredNetLeakyRelu = 2
    DualPredNetV1 = 3
    DarkPredNetV2 = 4

def model_factory(model_type, **args):
    def time_distribute_loss(errors):
        from tensorflow.keras.layers import TimeDistributed, Flatten, Dense
        layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; 
                                                        # "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
        layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
        time_loss_weights = 1./ (args['nt'] - 1) * np.ones((args['nt'],1))  # equally weight all timesteps except the first
        time_loss_weights[0] = 0
        errors_by_time = TimeDistributed(Dense(1, trainable=False), 
                                         weights=[layer_loss_weights, np.zeros(1)], 
                                         trainable=False)(errors)  # calculate weighted error by layer
        errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
        final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
        return final_errors
    
    
    if model_type == ModelType.PredNetOriginal:
        # Requires: sz, output_mode, num_gpus
        n_channels, im_height, im_width = (1, args['sz'], args['sz'])
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
                                  output_mode=args['output_mode'])
        prednet = PredNet(prednet_cell)
        errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers) 
        final_errors = time_distribute_loss(errors)
        model = Model(inputs=inputs, outputs=final_errors)
        if args['num_gpus'] > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=args.num_gpus)
        optimizer=SGD()
        model.compile(loss='mean_absolute_error', optimizer=optimizer) 
        
        return model
    
    elif model_type == ModelType.PredNetLeakyRelu:
        # Requires: sz, output_mode, num_gpus, output_shape
        n_channels, im_height, im_width = (1, args['sz'], args['sz'])
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
                                   output_mode=args['output_mode'],
                                   A_activation='LeakyReLU')
        prednet = PredNet(prednet_cell, stateful=args['stateful'])
        
        nt = 1 if args['output_mode'] == 'prediction' else args['nt']
        inputs = tf.keras.Input(batch_shape=(args['bs'], nt) + input_shape)
        if args['output_mode'] == 'prediction':
            prediction = prednet(inputs)
            model = Model(inputs=inputs, outputs=prediction)
        elif args['output_mode'] == 'error':
            errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers) 
            final_errors = time_distribute_loss(errors)
            model = Model(inputs=inputs, outputs=final_errors)
            
        if args['num_gpus'] > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=args.num_gpus)
            
        optimizer=SGD()
        #optimizer=Adam(lr=1e-3/2)
        model.compile(loss='mean_absolute_error', optimizer=optimizer) 
        return model
        
    elif model_type == ModelType.DualPredNetV1:
        # Requires: sz, output_mode, num_gpus, output_shape
        n_channels, im_height, im_width = (1, args['sz'], args['sz'])
        input_shape = (im_height, im_width, n_channels)
        stack_sizes = (n_channels, 64, 128, 256)
        R_stack_sizes = stack_sizes
        A_filt_sizes = (3, 3, 3)
        Ahat_filt_sizes = (3, 3, 3, 3)
        R_filt_sizes = (3, 3, 3, 3) 
        prednet_cell = DualPredNetCellV1(stack_sizes=stack_sizes,
                                   R_stack_sizes=R_stack_sizes,
                                   A_filt_sizes=A_filt_sizes,
                                   Ahat_filt_sizes=Ahat_filt_sizes,
                                   R_filt_sizes=R_filt_sizes, 
                                   output_mode=args['output_mode'],
                                   A_activation='LeakyReLU')
        prednet = PredNet(prednet_cell)
        
        inputs = tf.keras.Input(shape=(args['nt'],) + input_shape)
        if args['output_mode'] == 'prediction':
            prediction = prednet(inputs)
            model = Model(inputs=inputs, outputs=prediction)
        elif args['output_mode'] == 'error':
            errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers) 
            final_errors = time_distribute_loss(errors)
            model = Model(inputs=inputs, outputs=final_errors)
            
        if args['num_gpus'] > 1:
            model = tf.keras.utils.multi_gpu_model(model, args['num_gpus'], False)
            
        optimizer=SGD()
        model.compile(loss='mean_absolute_error', optimizer=optimizer) 
        return model
        
    elif model_type == ModelType.DarkPredNetV2:
        # Requires: sz, output_mode, num_gpus, output_shape
        n_channels, im_height, im_width = (1, args['sz'], args['sz'])
        input_shape = (im_height, im_width, n_channels)
        stack_sizes = (n_channels, 64, 128, 256)
        R_stack_sizes = stack_sizes
        A_filt_sizes = (3, 3, 3)
        Ahat_filt_sizes = (3, 3, 3, 3)
        R_filt_sizes = (3, 3, 3, 3)
        prednet_cell = DarkPredNetCellV2(stack_sizes=stack_sizes,
                                   R_stack_sizes=R_stack_sizes,
                                   A_filt_sizes=A_filt_sizes,
                                   Ahat_filt_sizes=Ahat_filt_sizes,
                                   R_filt_sizes=R_filt_sizes, 
                                   output_mode=args['output_mode'],
                                   A_activation='LeakyReLU')
        prednet = PredNet(prednet_cell)
        
        inputs = tf.keras.Input(shape=(args['nt'],) + input_shape)
        if args['output_mode'] == 'prediction':
            prediction = prednet(inputs)
            model = Model(inputs=inputs, outputs=prediction)
        elif args['output_mode'] == 'error':
            errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers) 
            final_errors = time_distribute_loss(errors)
            model = Model(inputs=inputs, outputs=final_errors)
            
        if args['num_gpus'] > 1:
            model = tf.keras.utils.multi_gpu_model(model, gpus=args['num_gpus'])
            
        optimizer=SGD()
        model.compile(loss='mean_absolute_error', optimizer=optimizer) 
        return model
    
    else:
        raise NotImplementedError
        
        
        
        

