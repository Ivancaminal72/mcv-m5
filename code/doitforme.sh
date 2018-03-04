#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_0.py -e fridaynight_belgium_0
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_1.py -e fridaynight_belgium_1
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/fridaynight_lamlam_0.py -e fridaynight_lamlam_0_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_1.py -e fridaynight_lamlam_1_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0.py -e fridaynight_vgg_0
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1.py -e fridaynight_vgg_1
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_ImageNet.py -e Belgium_vgg_ImageNet
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_TT100K.py -e Belgium_TT100K
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_spp_no_resize.py -e sabato_lamlam_spp_no_resize
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_spp.py -e sabato_lamlam_spp_resize244
CUDA_VISIBLE_DEVICES=0 python train.py -c config/KITTI_classif_ImageNet.py -e KITTI_vgg_ImageNet
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_lamlam_spp_no_resize_fine_tunning.py -e lamlam_spp_fine_tuning_Belgium_no_resize
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_lamlam_spp_fine_tunning.py -e lamlam_spp_fine_tuning_Belgium_224
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_lamlam_no_resize_fine_tunning.py -e lamlam_fine_tuning_Belgium_no_resize
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_lamlam_fine_tunning -e lamlam_fine_tuning_Belgium_224
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/KITTI_classif_TT100K.py -e KITTI_TT100K 
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_1.py -e fridaynight_lamlam_1_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0std.py -e sabato_morning_vgg0_more_patient_with_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1std.py -e sabato_morning_vgg1_more_patient_with_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0.py -e sabato_morning_vgg0_more_patient_without_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1.py -e sabato_morning_vgg1_more_patient_without_std

