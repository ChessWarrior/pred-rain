# Create a new model

## 1. Import the `Base_Model` from `base_model.py` and inherit it

```python
from .base_model import Base_Model
class Your_Fancy_Model(Base_Model):
	"""docstring for Your_Fancy_Model"""
	def __init__(self, args):
		super(Your_Fancy_Model, self).__init__(args)
		self.args = args
	# You need to implement build() fit() predict() and restore() functions
```

## 2. Add the script the start training your model

```python

def train(args):
	dr = np_data_reader(args)
	sess = tf.InteractiveSession()
	model = Your_Fancy_Model(args)
	X = tf.placeholder(shape=[None, 31, 501, 501, 1], dtype=tf.float32)
	y = tf.placeholder(shape=[None, 30, 501, 501, 1], dtype=tf.float32)
	model.build(X, y)

	model.fit(sess, dr, args.model_path)
```