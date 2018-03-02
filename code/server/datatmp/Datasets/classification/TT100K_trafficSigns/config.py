import imp
import os

# Dataset
dataset_name    = 'TT100K_trafficSigns'           # Name of the dataset
n_channels      = 3               # Number of channels of the images
color_mode      = 'rgb'           # Number of channels of the images [rgb, grayscale]
void_class      = []              # Labels of the void classes (It should be greather than the number of classes)
img_shape       = (64, 64)      # Shape of the input image (Height, Width)
n_images_train  = 16527            # Number of training images
n_images_valid  = 1644             # Number of validation images
n_images_test   = 8190             # Number of testing images
class_mode      = 'categorical'   # {'categorical', 'binary', 'sparse', 'segmentation', None}:


# Normalization constants
rgb_mean        = [0, 0, 0]        # Pixel mean to be substracted
rgb_std         = [1, 1, 1]        # Pixel std to be divided
rgb_rescale     = 1/255.           # Scalar to divide and set range 0-1

# Classes
classes = {0:'i2',
           1:'i4',
           2:'i5',
           3:'il100',
           4:'il60',
           5:'il80',
           6:'io',
           7:'ip',
           8:'p10',
           9:'p11',
           10:'p12',
           11:'p19',
           12:'p23',
           13:'p26',
           14:'p27',
           15:'p3',
           16:'p5',
           17:'p6',
           18:'pg',
           19:'ph4',
           20:'ph4.5',
           21:'ph5',
           22:'pl100',
           23:'pl120',
           24:'pl20',
           25:'pl30',
           26:'pl40',
           27:'pl5',
           28:'pl50',
           29:'pl60',
           30:'pl70',
           31:'pl80',
           32:'pm20',
           33:'pm30',
           34:'pm55',
           35:'pn',
           36:'pne',
           37:'po',
           38:'pr40',
           39:'w13',
           40:'w32',
           41:'w55',
           42:'w57',
           43:'w59',
           44:'wo'}


n_classes = len(classes) - len(void_class) # Not including the void classes
