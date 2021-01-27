#!/bin/bash
ls -d $2/*freiburg* | while read data_root
do
	$1/test_online ${data_root} ${data_root}/online.yaml
	echo ${data_root}
done


