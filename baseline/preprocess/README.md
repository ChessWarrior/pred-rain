# Usage for data converter 

## 1. Convert data to `.tfrecords`

```bash
python cvt2tfrecord.py --data_dir ../data --records data
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

check out the `data_reader.py`