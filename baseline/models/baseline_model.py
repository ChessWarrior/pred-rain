# we use conv lstm as our baseline model
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
#from cell import ConvLSTMCell
# tf.nn.dynamic_rnn

class Base_Model(object):
	"""Base_Model should only be inherited not to be directly used"""
	def __init__(self, args):
		self.args = args

	def build(self, X_train, y_train):
		raise NotImplementedError('build function is not implemented')

	def fit(self, sess, X_train, y_train):
		raise NotImplementedError('fit function not implemented')

	def predict(self, sess, X):
		raise NotImplementedError('predict function is not implemented')

	def restore(self, sess, path):
		raise NotImplementedError('restore function is not implemented')

class ConvLSTM_Model(Base_Model):
	'''
		ConvLSTM_Model 
	'''
	def __init__(self, args):
		super(ConvLSTM_Model, self).__init__(args)
		self.args = args
		self.saver = tf.train.Saver()
		
	def build(self, X, y):
		# TODO: check self.args.mode to decide which mode to build
		# X.shape == (None, 31, 501, 501, 1)
		# y.shape == (None, 30, 501, 501, 1)
		#filters = 256
		#kernel = (3, 3)
		#lstmcell = ConvLSTMCell(X.shape[2:4], filters, kernel)
		#outputs, state = tf.nn.dynamic_rnn(lstmcell, X, dtype=X.dtype)
		self.net = InputLayer(X, name='input')
		self.net = ConvLSTMLayer(self.net, n_steps=30, name='ConvLSTMLayer1')
		# self.outputs.shape == (None, 30, 501, 501, 1)
		self.outputs = self.net.outputs

		self.loss = tf.nn.l2_loss(y - self.outputs)
		self.optim = tf.train.AdamOptimizer()
		self.train_op = self.optim.minimize(self.loss)

	def fit(self, sess, X_train, y_train, save_path):
		sess.run(tf.global_variables_initializer())

		for epoch in range(self.args.epochs):
			if epoch + 1 % self.args.save_interval:
				# TODO: save the model
				pass

			if epoch + 1 % self.args.log_interval:
				# TODO: print the needed log info
				pass


	def predict(self, sess, X):
		# TODO: predict the (batch_size, 30, 501, 501, 1) prediction
		pass

	def restore(self, sess, path):
		# TODO: restore the model to sess
		self.saver.restore(sess, path)