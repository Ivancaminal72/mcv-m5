# Keras imports

from keras.preprocessing import image
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Flatten, Dropout, Input
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D,BatchNormalization,Conv2D,Concatenate
from keras import backend as K #$$
from keras import optimizers #$$

#from keras.utils.visualize_util import plot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np


from keras.layers.merge import concatenate
import math

# Paper: https://arxiv.org/pdf/1409.1556.pdf

def build_lamlam(img_shape=(3, 224, 224), n_classes=1000, n_layers=16, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):

      #I. image preprocessing
    x = img_shape[1]
    y = img_shape[2]
    # conv 2d layers
    activ1 = 'sigmoid'
    activ2 = 'relu'
    filt_size1_1 = 3
    filt_size1_2 = 5
    filt_size2_1 = 15
    filt_size2_2 = 19

    output1_conv = 8
    output2_conv = 64
    # polling layers
    final_x = 12
    final_y = 12
    stride1_1 = 4
    stride2_1 = 4
    stride1_2 = 4
    stride2_2 = 4

    # dense layer (fully connected)
    output_fc = 512
    activ_fc = 'sigmoid'
    # optimization process
    lr  = 1


    # Calculating the pool pool_size1
    out1_after_conv = [np.ceil(np.divide(x,stride1_1,dtype='float')),math.ceil(np.divide(y,stride1_2,dtype='float'))]
    out2_after_conv = [np.ceil(np.divide(x,stride2_1,dtype='float')),math.ceil(np.divide(y,stride2_2,dtype='float'))]
    pool_size1_1 = np.int(np.ceil(np.divide(out1_after_conv[0],final_x)))
    pool_size1_2 = np.int(np.ceil(np.divide(out1_after_conv[1],final_y)))
    pool_size2_1 = np.int(np.ceil(np.divide(out2_after_conv[0],final_x)))
    pool_size2_2 = np.int(np.ceil(np.divide(out2_after_conv[1],final_y)))

    input_shape = Input(shape=img_shape)
    # fine-tunining- freeze-layers
    train_layer_flag = True

    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
 	    train_layer_flag = False
    # tower_1 : texture - small filter

    tower_1 = Conv2D(output1_conv,(filt_size1_1,filt_size1_2),strides = (stride1_1,stride1_2),padding='same',activation = activ1,trainable=train_layer_flag,name='tower1_conv2')(input_shape)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = MaxPooling2D((pool_size1_1,pool_size1_2),trainable=train_layer_flag, name='tower1_pool')(tower_1)
    # tower_2 : behavair- organization - bigger filter
    tower_2 = Conv2D(output2_conv,(filt_size2_1,filt_size2_2),strides = (stride2_1,stride2_2),padding='same',activation = activ2,trainable=train_layer_flag,name='tower2_conv2')(input_shape)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = MaxPooling2D((pool_size2_1,pool_size2_2),name='tower2_pool')(tower_2)

    # merging towers
    merged = concatenate([tower_1,tower_2],axis=-1)
    merged = Flatten()(merged)
    # dense layers
    out = Dense(output_fc,activation=activ_fc,name='fc')(merged)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(n_classes,activation='softmax',name='predictions')(out)
	# Build a new model ( input the sane, output in the x layer)
    model = Model(input=input_shape, output=out)
    #if freeze_layers_from is not None:
    #    if freeze_layers_from == 'base_model':
    #        print ('   Freezing base model layers')
    #        for layer in merged.layers:
    #            layer.trainable = False
    #model.summary()
	# dont train the layers in base mode:l
    #model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adadelta(lr=lr), metrics=['accuracy'])

    return model
