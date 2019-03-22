#!/bin/sh
ssh="ssh ltin@bender.engr.ucr.edu 'cd functions/edge_detection/ &&
./edge $1.png && exit'"
scpFromServerToBender="scp ./public/rest/edge_detection/$1.png ltin@bender.engr.ucr.edu:functions/edge_detection/$1.png"
scpFromBenderToServer="scp ltin@bender.engr.ucr.edu:functions/edge_detection/$1_gpu.png ./public/rest/edge_detection/$1_gpu.png"

eval $scpFromServerToBender &&
eval $ssh &&
eval $scpFromBenderToServer