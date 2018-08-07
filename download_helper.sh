#!/bin/bash
# Start downloads and put them to background, accepting download URLs from stdin.
# Argument: The index the counter initializes to. Useful when the first files already exist.
# Instruction: Run this program in bash, passing in the index of the first file 
#	you wish the download starts at, e.g. 1, as the first argument (e.g. bash FILENAME.sh 1). Then open the competitions webpage: 
#	https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.79911bbc36Z3q3&raceId=231662 
#	and log in so you see available download links. Click on them to start 
#	download in browser and cancel to obtain download history in your 
#	download manager (press CTRL+J to open download managerin Chrome). 
#	Right-click on download link to copy it to clipboard and paste them 
#	in this program seperately. Use `tail -f log_$FN.log` to monitor 
#	download progress.
i=$1
while [ $i -le 10 ]
do
	read -p "URL to file $i:" URL 
	echo
	FN=${URL:85:22}
	nohup wget -c -O $FN $URL &> "log_$FN.log" &
	
	let i+=1
done
