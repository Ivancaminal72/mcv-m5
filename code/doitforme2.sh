#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_0.py -e sabato_morning_Belgium0_no_debug
CUDA_VISIBLE_DEVICES=0 python train.py -c config/Belgium_classif_1.py -e sabato_morning_Belgium1_no_debug
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_spp_no_resize.py -e sabato_lamlam_spp_no_resize
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_lamlam_spp.py -e sabato_lamlam_spp_resize244
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0std.py -e sabato_morning_vgg0_more_patient_with_std
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1std.py -e sabato_morning_vgg1_more_patient_with_std
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_0.py -e sabato_morning_vgg0_more_patient_without_std
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif_vgg_1.py -e sabato_morning_vgg1_more_patient_without_std



