# we use conv lstm as our baseline model
import tensorflow as tf
from cell import ConvLSTMCell
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
	def __init__(self, args):
		super(ConvLSTM_Model, self).__init__(args)
		self.args = args
		
	def build(self, X, y):
		# TODO: check self.args.mode to decide which mode to build
		# X.shape == (None, 31, 501, 501, 1)
		# y.shape == (None, 30, 501, 501, 1)
		filters = 256
		kernel = (3, 3)
		lstmcell = ConvLSTMCell(X.shape[2:4], filters, kernel)
		outputs, state = tf.nn.dynamic_rnn(lstmcell, X, dtype=X.dtype)
		# TODO: implement the conv-lstm model here


	def fit(self, sess, X_train, y_train, save_path):
		# TODO: training script
		pass

	def predict(self, sess, X):
		# TODO: predict the (batch_size, 30, 501, 501, 1) prediction
		pass

	def restore(self, sess, path):
		# TODO: restore the model to sess
		pass