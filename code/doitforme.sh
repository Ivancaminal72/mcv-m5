#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py -c config/fcn8_cityscapes.py -e fcn8_cityscapes_with_synthia2
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/fcn8_synthia_test.py -e synthia2
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_unet_da2.py -e unet_camvid_da_ch_shift4
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_unet_test.py -e unet_better
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_unet_test.py -e unet_rare_camvid

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_unet_da.py -e unet_da_reg
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/fcn8_synthia_test.py -e synthia2
#CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_unet_common_freq.py -e unet_common_camvid
