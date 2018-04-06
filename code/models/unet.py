# Keras imports
from keras.models import *
from keras.layers import *
import numpy as np
import keras.backend as K
# Two Versions of unet (zero padding vs mirror padding with croping)
# it is possible to set the type you want from the input : paddind = {'mirror', 'zero'}
# -*- coding: utf-8 -*-
#from __future__ import absolute_import
from keras.engine import Layer, InputSpec
from keras.layers.merge import _Merge

# imports for backwards namespace compatibility

class MirrorPadding2(Layer):
    '''Cropping layer for 2D input (e.g. picture).

    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_crop, second_axis_to_crop)

    # Output shape
        4D tensor with shape:
        (samples, depth, first_cropped_axis, second_cropped_axis)

    # Arguments
        padding: tuple of tuple of int (length 2)
            How many should be trimmed off at the beginning and end of
            the 2 padding dimensions (axis 3 and 4).
    '''
    input_ndim = 4

    def __init__(self, extending=((1,1),(1,1)), dim_ordering=K.image_dim_ordering(), **kwargs):
        super(MirrorPadding2, self).__init__(**kwargs)
        assert len(extending) == 2, 'cropping mus be two tuples, e.g. ((1,1),(1,1))'
        assert len(extending[0]) == 2, 'cropping[0] should be a tuple'
        assert len(extending[1]) == 2, 'cropping[1] should be a tuple'
        self.extending = tuple(extending)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        return input_shape
        # if self.dim_ordering == 'th':
        #
        #     return (input_shape[0],
        #             input_shape[1] ,
        #             input_shape[2] ,
        #             input_shape[3] + self.extending[1][0] + self.extending[1][1])
        # elif self.dim_ordering == 'tf':
        #     return (input_shape[0],
        #             input_shape[1] + self.extending[0][0] + self.extending[0][1],
        #             input_shape[2] + self.extending[1][0] + self.extending[1][1],
        #             input_shape[3])
        # else:
        #     raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x):
        """
        width, height = self.output_shape()[2], self.output_shape()[3]
        width_crop_left = self.cropping[0][0]
        height_crop_top = self.cropping[1][0]

        return x[:, :, width_crop_left:width+width_crop_left, height_crop_top:height+height_crop_top]
        """
        #cropping the image from the zero padding
        if self.dim_ordering == 'th':
            x = x[:, :, self.extending[0][0]:x.shape[2]-self.extending[0][1], self.extending[1][0]:x.shape[3]-self.extending[1][1]]
        else:
            x = x[:, self.extending[0][0]:x.shape[1]-self.extending[0][1], self.extending[1][0]:x.shape[2]-self.extending[1][1],:]
        print(x)
        up = self.crop(x,'up')
        dn = self.crop(x,'dn')

        if self.dim_ordering == 'th':
            x = K.concatenate((K.reverse(up,axes=2),x,K.reverse(dn,axes=2)),axis=2)
        else:
            x = K.concatenate((K.reverse(up,axes=1),x,K.reverse(dn,axes=1)),axis=1)
        l = self.crop(x,'l')
        r = self.crop(x,'r')
        if self.dim_ordering == 'th':
            x = K.concatenate((K.reverse(l,axes=3),x,K.reverse(r,axes=3)),axis=3)
        else:
            x = K.concatenate((K.reverse(l,axes=2),x,K.reverse(r,axes=2)),axis=2)
        print(x)
        return x

    def crop(self,x,dire = 'up'):
        if self.dim_ordering == 'th':
            if dire == 'up':
                out = x[:, :, 0:self.extending[0][0], :]
            elif dire == 'dn':
                out = x[:, :, x.shape[2]-self.extending[0][1]:x.shape[2], :]
            elif dire == 'l':
                out = x[:, :, :, 0:self.extending[1][0]]
            elif dire == 'r':
                out = x[:, :, :, x.shape[3]-self.extending[1][1]:x.shape[3]]
            else:
                print('No extention type')

        elif self.dim_ordering == 'tf':
            if dire == 'up':
                out = x[:, 0:self.extending[0][0],:, :]
            elif dire == 'dn':
                out = x[:, x.shape[1]-self.extending[0][1]:x.shape[1],:, :]
            elif dire == 'l':
                out = x[:, :, 0:self.extending[1][0],:]
            elif dire == 'r':
                out = x[:, :,  x.shape[2]-self.extending[1][1]:x.shape[2],:]
            else:
                print('No extention type')
        return out


    def get_config(self):
        config = {'extending': self.extending}
        base_config = super(MirrorPadding2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_unet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0., init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None,padding = 'mirror'):
    if padding is 'mirror':
        model = build_unet_mirror(img_shape, nclasses, l2_reg,init, path_weights,load_pretrained)
    elif padding is 'zero':
        model = build_unet_zero(img_shape, nclasses, l2_reg,init, path_weights,load_pretrained)
    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)
    return model



def build_unet_mirror(img_shape=(368, 464, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None):
    #input shape has to be at least : 16x-124 ( because of the conv boundries loss and because of the pooling ( if it is an odd number we are lossing a pix))
	# https://github.com/zhixuhao/unet/blob/master/unet.py
    img_input = Input(shape=img_shape)

    # 4 dim tenssor [batch, height,width, channel]
    test_input_size = (np.asarray(img_shape,dtype=np.float32) + 124.)/16.
    if test_input_size[0].round() == test_input_size[0] and test_input_size[1].round() == test_input_size[1]:
        print('input shape is valid to unet mirror with valid padding')

    else:
        print('input shape is NOT valid to unet mirror with valid padding!!')
    frame_size = [[92,92],[92,92]]
    # If not using zeropadding to initialize the new sizze - it doesnt updates
    i1=ZeroPadding2D(padding=frame_size)(img_input)
    img_input_mirror_pad = MirrorPadding2(extending=frame_size)(i1)
    #paddings = K.tf.constant([[0,0],[92,92,], [92,92],[0,0]])
    #img_input = K.tf.pad(img_input, paddings, "REFLECT")
    #print(type(img_input))
    # change it from zeropadding to mirror!!!
    #print "img_input_mirror_pad shape:",img_input_mirror_pad.shape

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(img_input_mirror_pad)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    crop1 = Cropping2D(cropping=((88,88),(88,88)))(conv1)
    #print "crop1 shape:",crop1.shape
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #print "pool1 shape:",pool1.shape
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    crop2 = Cropping2D(cropping=((40,40),(40,40)))(conv2)
    #print "crop2 shape:",crop2.shape
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #print "pool2 shape:",pool2.shape
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    #print "conv3 shape:",conv3.shape
    crop3 = Cropping2D(cropping=((16,16),(16,16)))(conv3)
    #print "crop3 shape:",crop3.shape
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    #print "conv4 shape:",conv4.shape
    crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
    #print "crop4 shape:",crop4.shape
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #print "pool4 shape:",pool4.shape
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
    #print "conv5 shape:",conv5.shape
    drop5 = Dropout(0.5)(conv5)

    #up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',data_format)(drop5)
    #print "up6 shape:",up6.shape
    #up6 = Conv2DTranspose(256, 3, 3, )(drop5)
    #print "drop5 shape:",drop5.shape
    up6 = UpSampling2D(size=(2,2))(conv5)
    #print "up6 shape:",up6.shape
    up6 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(conv5))
    #print "up6 shape:",up6.shape
    #print "crop4 shape:",crop4.shape
    merge6 = concatenate([crop4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
    #print "conv6 shape:",conv6.shape
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #print "up7 shape:",up7.shape
    merge7 = concatenate([crop3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #print "up8 shape:",up8.shape
    merge8 = concatenate([crop2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #print "up9 shape:",up9.shape
    merge9 = concatenate([crop1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
    o = Conv2D(nclasses, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	#o = Conv2D(nclasses, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	#o = Conv2D(1, 1, activation = 'sigmoid')(conv9) # last classification layer in the original article --
    curlayer_output_shape = Model(input = img_input, output = o).output_shape
    if K.image_dim_ordering() == 'tf':
        outputHeight = curlayer_output_shape[1]
        outputWidth = curlayer_output_shape[2]
    else:
        outputHeight = curlayer_output_shape[2]
        outputWidth = curlayer_output_shape[3]
    o = Reshape(target_shape=(outputHeight * outputWidth, nclasses))(o)
    o = Activation('softmax')(o)
    model = Model(input =img_input,output= o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    return model

def build_unet_zero(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None):


	# https://github.com/zhixuhao/unet/blob/master/unet.py
	img_input = Input(shape=img_shape)

	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_input)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
   	merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	o = Conv2D(nclasses, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

	curlayer_output_shape = Model(input = img_input, output = o).output_shape
	if K.image_dim_ordering() == 'tf':
		outputHeight = curlayer_output_shape[1]
		outputWidth = curlayer_output_shape[2]
	else:
		outputHeight = curlayer_output_shape[2]
		outputWidth = curlayer_output_shape[3]

	o = Reshape(target_shape=(outputHeight * outputWidth, nclasses))(o)

	o = Activation('softmax')(o)
	model = Model(input =img_input,output= o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	# Freeze some layers
	if freeze_layers_from is not None:
		freeze_layers(model, freeze_layers_from)


    	return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


if __name__ == '__main__':
    input_shape = [224, 224, 3]
    print (' > Building')
    model = build_unet(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
