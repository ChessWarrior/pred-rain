# baseline model using ConvLSTM 

## 1. Put the data into `data/` folder

the `data/` should be the following format
```
data/
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

## 2. Create some folders

```bash
mkdir logs
mkdir pretrained
mkdir samples
```

## 3. Start training

```bash
python main.py
```