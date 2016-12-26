#!/bin/bash -e
th train.lua --mode test --feature_type ours --gpuDevice 1 --use_bn true --net regular_net --batchsize 64 --datasource kgs --num_forward_models 2048 --nthread 4 --alpha 0.01 --epoch_size 128000 --data_augmentation

