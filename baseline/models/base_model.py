import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

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
