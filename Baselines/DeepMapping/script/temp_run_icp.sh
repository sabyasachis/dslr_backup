#!/bin/bash

# path to dataset
DATA_DIR=../data/2D/10-00-14-P1-6-auto-ccw_5loops_0.6_no_numba_every_tenth_scan/
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=dhiraj_with_icp
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=plane

python incremental_icp.py --name $NAME -d $DATA_DIR -m $METRIC 
