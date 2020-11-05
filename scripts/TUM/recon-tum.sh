#!/bin/bash
ls -d $2/*freiburg* | while read data_root
do
	if [[ $3 == 'online' ]]; then
		$1/test_online ${data_root} /home/liulei/codes/GraphFusion/test/test_online/online_pnp.yaml
	else
		$1/test_multi ${data_root} $3
	fi
	echo ${data_root}
done


