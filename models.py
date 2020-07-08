from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *  
from DUAttention import *
kinit = 'he_normal'  

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input,Lambda
from keras.layers import MaxPooling2D,Concatenate,UpSampling2D, add, Dense, Multiply
from keras.layers import merge, concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D,BatchNormalization
from keras.models import Model


def conv2d_bn(x,filters,num_row,num_col,padding='same',stride=1,dilation_rate=1,relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride,stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    if relu:    
        x = Activation("relu")(x)
    return x

def UnetConv2D(input, outdim, is_batchnorm, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x   

def CA_UNet(input_size):
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    #print('conv3', K.int_shape(conv3)[0])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
    #print('conv4', K.int_shape(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)
    print('conv5', conv5.shape)

    # up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5)
    #print('up6', K.int_shape(up6))
    #up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4], axis=3)
    up6 = Concatenate(axis=3)([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4])
    # cat6 = Concatenate(axis=3)([up6, conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up6) # rrrrrrrrr
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv6)

    #sa_1 = PAM()(conv6)
    sc_1 = CAM()(conv6)#, '_sc1')
    #dual_1 = Add()([sa_1, sc_1])
    #print('dual_1', K.int_shape(sc_1))

    # up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sc_1)
    #print('up7', K.int_shape(up7))
    #up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(sc_1), conv3], axis=3)
    up7 = Concatenate(axis=3)([Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sc_1), conv3])
    #cat7 = Concatenate(axis=3)([up7, conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7) #rrrrrr
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

    #sa_2 = PAM()(conv7)#PAM(conv7, 128, '_sa2')
    sc_2 =  CAM()(conv7)#,( '_sc2')
    #dual_2 = Add()([sa_2, sc_2])

    #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_2), conv2], axis=3)
    up8 = Concatenate(axis=3)([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sc_2), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv8)

    #sa_3 = PAM()(conv8)
    sc_3 = CAM()(conv8)
    #dual_3 = Add()([sa_3, sc_3])

    ###up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_3), conv1], axis=3)
    up9 = Concatenate(axis=3)([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sc_3), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
    #conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dual_3)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)
    #print('conv10 zoumei')

    model = Model(inputs=inputs, outputs=conv10)
    #print('zou meizou')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def PA_UNet(input_size):
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    #print('conv3', K.int_shape(conv3)[0])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
    #print('conv4', K.int_shape(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)
    print('conv5', conv5.shape)

    # up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5)
    #print('up6', K.int_shape(up6))
    #up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4], axis=3)
    up6 = Concatenate(axis=3)([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4])
    # cat6 = Concatenate(axis=3)([up6, conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up6) # rrrrrrrrr
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv6)

    sa_1 = PAM()(conv6)
    #sc_1 = CAM()(conv6)#, '_sc1')
    #dual_1 = Add()([sa_1, sc_1])
    #print('dual_1', K.int_shape(sc_1))

    # up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sc_1)
    #print('up7', K.int_shape(up7))
    #up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(sc_1), conv3], axis=3)
    up7 = Concatenate(axis=3)([Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sa_1), conv3])
    #cat7 = Concatenate(axis=3)([up7, conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7) #rrrrrr
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

    sa_2 = PAM()(conv7)#PAM(conv7, 128, '_sa2')
    #sc_2 =  CAM()(conv7)#,( '_sc2')
    #dual_2 = Add()([sa_2, sc_2])

    #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_2), conv2], axis=3)
    up8 = Concatenate(axis=3)([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sa_2), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv8)

    sa_3 = PAM()(conv8)
    #sc_3 = CAM()(conv8)
    #dual_3 = Add()([sa_3, sc_3])

    ###up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_3), conv1], axis=3)
    up9 = Concatenate(axis=3)([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(sa_3), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
    #conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dual_3)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)
    #print('conv10 zoumei')

    model = Model(inputs=inputs, outputs=conv10)
    #print('zou meizou')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def DUA_UNet(input_size):
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    #print('conv3', K.int_shape(conv3)[0])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
    #print('conv4', K.int_shape(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)
    print('conv5', conv5.shape)

    # up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5)
    #print('up6', K.int_shape(up6))
    #up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4], axis=3)
    up6 = Concatenate(axis=3)([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4])
    # cat6 = Concatenate(axis=3)([up6, conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up6) # rrrrrrrrr
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv6)

    sa_1 = PAM()(conv6)
    sc_1 = CAM()(conv6)#, '_sc1')
    dual_1 = Add()([sa_1, sc_1])
    #print('dual_1', K.int_shape(sc_1))

    up7 = Concatenate(axis=3)([Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_1), conv3])
    #cat7 = Concatenate(axis=3)([up7, conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7) #rrrrrr
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

    sa_2 = PAM()(conv7)#PAM(conv7, 128, '_sa2')
    sc_2 =  CAM()(conv7)#,( '_sc2')
    dual_2 = Add()([sa_2, sc_2])

    #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_2), conv2], axis=3)
    up8 = Concatenate(axis=3)([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_2), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv8)

    sa_3 = PAM()(conv8)
    sc_3 = CAM()(conv8)
    dual_3 = Add()([sa_3, sc_3])

    ###up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_3), conv1], axis=3)
    up9 = Concatenate(axis=3)([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(dual_3), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
    #conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dual_3)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)
    #print('conv10 zoumei')

    model = Model(inputs=inputs, outputs=conv10)
    #print('zou meizou')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

    
    
