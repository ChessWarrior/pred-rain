# we use conv lstm as our baseline model
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
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
		self.net = InputLayer(self.X, name='input')
		self.net = ConvLSTMLayer(self.net, n_steps=30, name='ConvLSTMLayer1')
		# self.outputs.shape == (None, 30, 501, 501, 1)
		self.outputs = self.net.outputs

		self.loss = tf.nn.l2_loss(self.y - self.outputs)
		self.optim = tf.train.AdamOptimizer()
		self.train_op = self.optim.minimize(self.loss)
		self.build_graph_flag = True
		self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
		print('Successfully built the model')

	def fit(self, sess, data_reader, save_path):
		sess.run(tf.global_variables_initializer())
		for epoch in range(self.args.epochs):
			print('[*] Epoch {0} train_step: {1} val_step: {2}'
				.format(epoch, data_reader.train_n_steps, data_reader.val_n_steps))
			for step in range(data_reader.train_n_steps):
				X_batch, y_batch = data_reader.read_train_data()
				loss_, __ = sess.run([self.loss, self.train_op], 
									feed_dict={self.X: X_batch, self.y: y_batch})
				print('Epoch {0} / {1} [{2}]: loss \t {3}'
					.format(epoch, self.args.epochs, step, loss_))

			val_loss = 0
			for step in range(data_reader.val_n_steps):
				X_batch, y_batch = data_reader.read_val_data()
				loss_ = sess.run(self.loss, feed_dict={self.X: X_batch, self.y: y_batch})
				val_loss += loss
			val_loss /= data_reader.val_n_steps
			print('Epoch {0} / {1} [val]: loss \t {2}'
				.format(epoch, self.args.epochs, val_loss))

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