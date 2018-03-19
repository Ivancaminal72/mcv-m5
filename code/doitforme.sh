#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_0.py -e fridaynight_belgium_0debug
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_1.py -e fridaynight_belgium_1
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam.py -e fridaynight_lamlam_0_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_1.py -e fridaynight_lamlam_1_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0.py -e fridaynight_vgg_0_debug
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1.py -e fridaynight_vgg_1
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_ImageNet.py -e Belgium_vgg_ImageNet
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_TT100K.py -e Belgium_TT100K
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_spp_no_resize.py -e sabato_lamlam_spp_no_resize
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection_wb.py -e udacity_wb
CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection_hsv_wb.py -e udacity_hsv_wb_debug
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection_hsv_wb_sat_shift.py -e udacity_hsv_wb_s_shift
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection_hsv_wb_sat_val_shift.py -e udacity_hsv_wb_sv_shift
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection_hsv_sat_shift.py -e udacity_hsv_s_shift
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_ssd.py -e tt100k_ssd
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_ssd.py -e udacity_ssd

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_detection_416.py -e yolo416_tt100k_full_metrics
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/KITTI_classif_ImageNet.py -e KITTI_vgg_ImageNet
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection.py -e yolo_udacity_prime1
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_lamlam_spp_fine_tunning.py -e lamlam_spp_fine_tuning_Belgium_224
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/KITTI_lamlam_spp_no_resize_fine_tunning.py -e lamlam_spp_fine_tuning_KITTI_no_resize_16_03
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_lamlam_fine_tunning -e lamlam_fine_tuning_Belgium_224
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/KITTI_classif_TT100K.py -e KITTI_vgg_TT100K
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_1.py -e fridaynight_lamlam_1_with_train
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0std.py -e sabato_morning_vgg0_more_patient_with_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1std.py -e sabato_morning_vgg1_more_patient_with_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0.py -e sabato_morning_vgg0_more_patient_without_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1.py -e sabato_morning_vgg1_more_patient_without_std
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam.py -e sabato_lamlam_no_resize__corrected
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_0.py -e sabato_lamlam_resize244_0__corrected
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_1.py -e sabato_lamlam_resize244_1__corrected
