#!/bin/bash
# Start the conversion of tfrecords from data directories with the range specified by arguments.
# Arguments: The range of the index of the data directories to be converted to tfrecords.
# Assumed folder structure:
# .
# ├── data
# │   ├── SRAD2018_TRAIN_001
# │   ├── SRAD2018_TRAIN_002
# │   ├── ...
# │   ├── tfrecords
# │   	  ├── train_1.tfrecords
# | 	  ├── train_2.tfrecords
# └── scripts
#     ├── convert_helper.sh
#     ├── cvt2tfrecord.py
#     ├── log (created automatically)
#
#	 Use `tail -f log/$i.tfrecords.log` to trace download progress.

i=$1
END=$2
while [ $i -le $END ]
do
	N_FORMATTED="$(printf %03i $i)"

	[[ -d log ]] || mkdir log
	[[ -d ../data/tfrecords ]] || mkdir ../data/tfrecords
	nohup python cvt2tfrecord.py --data_dir ../data/SRAD2018_TRAIN_$N_FORMATTED/ --records ../data/tfrecords/train_$i &> "log/$i.tfrecords.log" &

	let i+=1
done
