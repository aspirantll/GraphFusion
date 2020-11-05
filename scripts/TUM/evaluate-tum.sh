#!/bin/bash
ls -d $1/*freiburg* | while read data_root
do
	echo ${data_root}
	python evaluate_ate.py "${data_root}/groundtruth.txt" "${data_root}/$2_estimate_$3.txt"
done


