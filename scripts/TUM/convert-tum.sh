#!/bin/bash
ls -d $2/*freiburg* | while read data_root
do
	if [ ! -f "${data_root}/frames_0.yaml" ]
	then
		$1/dataConverter ${data_root} TUM
		echo ${data_root}
	fi
done


