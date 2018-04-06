#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_segmentation.py -e given_segmentaion_after_lorenzos_fix
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_with_camvid_weights.py -e pretrained_on_camvid
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/fcn8_image.py -e photo_fcn8
