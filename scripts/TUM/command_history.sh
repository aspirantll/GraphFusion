./associate-tum.sh /media/liulei/Data/dataset/TUM
./convert-tum.sh /home/liulei/codes/GraphFusion/build/bin/ /media/liulei/Data/dataset/TUM
./recon-tum.sh /home/liulei/codes/GraphFusion/build/bin /media/liulei/Data/dataset/TUM online
./evaluate-tum.sh /media/liulei/Data/dataset/TUM
python evaluate_ate.py /media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_360/groundtruth.txt /media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_360/online_estimate_5.txt 
python evaluate_ate.py /media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_desk/groundtruth.txt /media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_desk/online_estimate_5.txt
python evaluate_ate.py /media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_floor/groundtruth.txt /media/liulei/Data/dataset/TUM/rgbd_dataset_freiburg1_floor/estimate_5.txt
