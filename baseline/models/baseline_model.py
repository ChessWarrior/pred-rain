# we use conv lstm as our baseline model
import tensorflow as tf
from cell import ConvLSTMCell
# tf.nn.dynamic_rnn

class Base_Model(object):
	"""Base_Model should only be inherited not to be directly used"""
	def __init__(self, args):
		self.args = args

	def build_graph(self, X_train, y_train):
		raise NotImplementedError('The build function of your model is not implemented')

	def fit(self, sess, X_train, y_train):
		raise NotImplementedError('The fit function of your model is not implemented')

	def predict(self, sess, X):
		raise NotImplementedError('The predict function of your model is not implemented')


class ConvLSTM_Model(Base_Model):
	def __init__(self, args):
		super(ConvLSTM_Model, self).__init__(args)
		self.args = args
		
	def build_graph(self, X, y):
		# X.shape == (None, 31, 501, 501, 1)
		# y.shape == (None, 30, 501, 501, 1)
		filters = 256
		kernel = (3, 3)
		lstmcell = ConvLSTMCell(X.shape[2:4], filters, kernel)
		outputs, state = tf.nn.dynamic_rnn(lstmcell, X, dtype=X.dtype)

	def fit(self, sess, X_train, y_train):
		pass

	def predict(self, sess, X):
		pass