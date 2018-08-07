# Usage for data converter and reader

## 1. Convert data to `.tfrecords`

```python
from cvt2tfrecord import cvter
cvt = cvter('01.tfrecords', '../data/')
cvt.convert()
```

`data_dir` should be directory in the following format
```
data_dir/
	RAD_206482414212530/
		RAD_206482414212530_000.png
		...
		RAD_206482414212530_060.png
	RAD_206482414212531/
		RAD_206482414212531_000.png
		...
		RAD_206482414212531_060.png
	...
```

`records` define the name of the generated `.tfrecords` file

## 2. Read the data from `.tfrecords` file

Detailed usage is in the `usage.ipynb`

also in `baseline_model.py` you can see how I use this for the ConvLSTM Model