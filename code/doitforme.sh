#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_0.py -e fridaynight_belgium_0
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_1.py -e fridaynight_belgium_1
CUDA_VISIBLE_DEVICES=0 python train.py -c config/fridaynight_lamlam_0.py -e fridaynight_lamlam_0_with_train
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_1.py -e fridaynight_lamlam_1_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0.py -e fridaynight_vgg_0
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1.py -e fridaynight_vgg_1
