#!/bin/sh
ssh ltin@bender.engr.ucr.edu 'cd functions/edge_detection/ &&
./edge ron.png && exit'
scp ltin@bender.engr.ucr.edu:functions/edge_detection/ron_gpu.png ./functions/edge_detection/output/test.png