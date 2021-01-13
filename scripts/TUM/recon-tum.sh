#!/bin/bash
ls -d $2/*freiburg* | while read data_root
do
	$1/test_online ${data_root} /home/liulei/codes/GraphFusion/test/test_online/online_pnp.yaml #${data_root}/online.yaml
	echo ${data_root}
done


