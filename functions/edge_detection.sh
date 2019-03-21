#!/bin/sh
# now=$(date +"%T") &&
# echo $now > functions/edge_detection/timestamp.txt
ssh ltin@bender.engr.ucr.edu 'cd functions/edge_detection/ &&
./edge ron.png && exit'
scp ltin@bender.engr.ucr.edu:functions/edge_detection/ron_gpu.png ./functions/edge_detection/test.png