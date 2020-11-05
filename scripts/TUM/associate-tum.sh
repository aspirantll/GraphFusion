#!/bin/bash
ls -d $1/*freiburg* | while read data_root
do
	if [ ! -f "${data_root}/associate.txt" ]
	then
		python "./associate.py" "${data_root}/rgb.txt" "${data_root}/depth.txt" > "${data_root}/associate.txt"
		echo ${data_root}
	fi

done
