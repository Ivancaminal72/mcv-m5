#!/bin/bash

python statistics_segmantation.py camvid train
python statistics_segmantation.py camvid valid
python statistics_segmantation.py camvid test
python statistics_segmantation.py kitti train
python statistics_segmantation.py kitti valid
python statistics_segmantation.py polyps train
python statistics_segmantation.py polyps valid
python statistics_segmantation.py polyps test
python statistics_segmantation.py synthia_rand_cityscapes valid
python statistics_segmantation.py synthia_rand_cityscapes train
python statistics_segmantation.py synthia_rand_cityscapes test
python statistics_segmantation.py cityscapes valid
python statistics_segmantation.py cityscapes train
python statistics_segmantation.py cityscapes test
python statistics_segmantation.py pascal2012 valid
python statistics_segmantation.py pascal2012 train
python statistics_segmantation.py pascal2012 test
python statistics_segmantation.py synthia_dataset valid
python statistics_segmantation.py synthia_dataset train
python statistics_segmantation.py synthia_dataset test
