# we use conv lstm as our baseline model
import tensorflow as tf
from .base_model import Base_Model

class ConvLSTM_Model(Base_Model):
    '''
        ConvLSTM_Model 
    '''
    def __init__(self, args):
        super(ConvLSTM_Model, self).__init__(args)
        self.args = args
        self.build_graph_flag = False
        
    def build(self, X, y):
        # TODO: check self.args.mode to decide which mode to build
        self.X, self.y = X, y

        self.net = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=self.X.shape,
                                            padding='same', return_sequences=True)(self.X)
        self.net = tf.keras.layers.BatchNormalization()(self.net)
        
        self.net = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3),
                                            padding='same', return_sequences=True)(self.net)
        self.net = tf.keras.layers.BatchNormalization()(self.net)

        self.net = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3),
                                            padding='same', return_sequences=True)(self.net)
        self.net = tf.keras.layers.BatchNormalization()(self.net)

        self.net = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3),
                                            padding='same', return_sequences=True)(self.net)
        self.net = tf.keras.layers.BatchNormalization()(self.net)

        self.net = tf.keras.layers.Conv3D(filters=1, kernel_size=(2, 1, 1), 
                                        padding='valid', data_format='channels_last')(self.net)
        # the (2, 1, 1) conv kernel converts (None, 31, 501, 501, 1) to (None, 30, 501, 501, 1)
        # self.outputs.shape == (None, 30, 501, 501, 1)
        self.outputs = self.net

        self.loss = tf.nn.l2_loss(self.y - self.outputs)
        self.optim = tf.train.AdamOptimizer()
        self.train_op = self.optim.minimize(self.loss)
        self.build_graph_flag = True
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
        print('Successfully built the model')

    def fit(self, sess, data_reader, save_path):
        sess.run(tf.global_variables_initializer())
        for epoch in range(self.args.epochs):
            print('[*] Epoch {0} train_step: {1} val_step: {2}'.format(epoch, data_reader.train_n_steps, data_reader.val_n_steps))

            fd = data_reader.set_mode('train')
            for step in range(data_reader.train_n_steps):
                loss_, __ = sess.run([self.loss, self.train_op], feed_dict=fd)
                print('Epoch {0} / {1} [{2}]: loss \t {3}'.format(epoch, self.args.epochs, step, loss_))

            val_loss = 0
            fd = data_reader.set_mode('valid')
            for step in range(data_reader.val_n_steps):
                loss_ = sess.run(self.loss, feed_dict=fd)
                val_loss += loss
            val_loss /= data_reader.val_n_steps
            print('Epoch {0} / {1} [val]: loss \t {2}'.format(epoch, self.args.epochs, val_loss))

            if epoch + 1 % self.args.save_interval:
                self.saver.save(sess, self.args.model_path, global_step=epoch)

            if epoch + 1 % self.args.log_interval:
                # TODO: save one output(the required six frame) to the samples dir
                #output = sess.run(self.outputs, feed_dict={self.X: X_batch})
                pass

            print('[*] Epoch {0} fininshed!'.format(epoch))

    def predict(self, sess, X):
        # TODO: predict the (batch_size, 30, 501, 501, 1) prediction
        pass

    def restore(self, sess, path):
        if self.build_graph_flag is False:
            self.saver = tf.train.import_meta_graph(self.args.model_path)
        self.saver.restore(sess, path)
