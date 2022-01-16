#!/bin/bash

# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=temp
# path to dataset
# DATA_DIR=../data/2D/v1_pose0
DATA_DIR=../data/2D/10-00-14-P1-6-auto-ccw_5loops_0.6_no_numba
# training epochs
EPOCH=3000
# batch size
BS=32
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=20
# logging interval
LOG=10

### training from scratch
#python train_2D.py --name $NAME -d $DATA_DIR -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
INIT_POSE=../results/2D/10-00-14-P1-6-auto-ccw_5loops_0.6_no_numba_every_tenth_scan_with_icp/pose_est.npy
python train_2D.py --name $NAME -d $DATA_DIR -i $INIT_POSE -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG
