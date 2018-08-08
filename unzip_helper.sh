#!/bin/bash

for fn in *.zip
do 
	unzip $fn -d ../${fn:0:-4}
done
