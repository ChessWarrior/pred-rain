#!/bin/bash
# Start the conversion of tfrecords from data directories with the range specified by arguments.
# Arguments: $1~$2: The range of the index of the data directories to be converted to tfrecords.
# 		$3: Conversion mode. One of 'contigious' and skip
#		$4: nt / skip factor
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
# Examples: 	./convert_helper.sh 1 2 skip 10 5
#		./convert_helper.sh 1 2 contiguous 10 && tail -f log/1_contiguous_10.tfrecords.log

i=$1
END=$2
MODE=$3
NT=$4
STOP=$5

while [ $i -le $END ]
do
	N_FORMATTED="$(printf %03i $i)"

	[[ -d cvt_logs ]] || mkdir cvt_logs
	[[ -d ../data/tfrecords ]] || mkdir ../data/tfrecords
	
	LOG_FN=cvt_logs/"$i"_"$MODE"_"$NT".tfrecords.log
	nohup python cvt2tfrecord.py --data_dir ../data/SRAD2018_TRAIN_$N_FORMATTED/ --records ../data/tfrecords/train_"$i"_"$MODE"_"$NT" --mode=$MODE --nt=$NT -stop=$STOP &> $LOG_FN &
    
	let i+=1
done
