#!/bin/bash

# path to dataset
DATA_DIR=../data/2D/carla_static_set1_tenth_frame
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=carla_static_set1_tenth_frame_with_icp
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=plane

python incremental_icp.py --name $NAME -d $DATA_DIR -m $METRIC 
