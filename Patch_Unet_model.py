# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import *

def Path_Unet(input_shape=(512, 512, 3), classes=3):
    
    bakcbone = tf.keras.applications.VGG16(input_shape=(224, 224, 3))
    
    inputs = tf.keras.Input(input_shape)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)

    c10 = Convolution2D(classes, 1, 1)(c9)

    model = tf.keras.Model(inputs=inputs, outputs=c10)

    # model.get_layer("conv2").set_weights(bakcbone.get_layer("block1_conv2").get_weights())
    # model.get_layer("conv3").set_weights(bakcbone.get_layer("block2_conv1").get_weights())
    # model.get_layer("conv4").set_weights(bakcbone.get_layer("block2_conv2").get_weights())
    # model.get_layer("conv5").set_weights(bakcbone.get_layer("block3_conv1").get_weights())
    # model.get_layer("conv6").set_weights(bakcbone.get_layer("block3_conv3").get_weights())


    return model

#import cv2
#import numpy as np

#def pad(image, height, width):
#    def get_padding_size(image):
#        h, w, _ = image.shape
#        longest_edge = max(h, w)
#        top, bottom, left, right = (0, 0, 0, 0)
#        if h < longest_edge:
#            dh = longest_edge - h
#            top = dh // 2
#            bottom = dh - top
#        elif w < longest_edge:
#            dw = longest_edge - w
#            left = dw // 2
#            right = dw - left
#        else:
#            pass
#        return top, bottom, left, right

#    top, bottom, left, right = get_padding_size(image)
#    BLACK = [0, 0, 0]
#    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

#    return constant

#def crop_image(image, crop_size):
#        # print(image.shape)
#        crops = []
#        size_x = crop_size[0]
#        size_y = crop_size[1]

#        n_crops_x = np.ceil(image.shape[0] / float(size_x)) 
#        n_crops_y = np.ceil(image.shape[1] / float(size_y)) 

#        required_x = n_crops_x * size_x
#        required_y = n_crops_y * size_y

#        # print(n_crops_x, n_crops_y, required_x, required_y)
#        required_x = required_x - image.shape[0]
#        required_y = required_y - image.shape[1]

#        # print(n_crops_x, n_crops_y, required_x, required_y)
#        required_x /= 2.
#        required_y /= 2.

#        required_x = required_x
#        required_y = required_y

#        # print(n_crops_x, n_crops_y, required_x, required_y)

#        new_image = pad(image, required_x, required_y)
#        bordersize=10
#        border=cv2.copyMakeBorder(image, top=int(np.ceil(required_x)), bottom=int(np.floor(required_x)), left= int(np.ceil(required_y)), right=int(np.floor(required_y)), borderType= cv2.BORDER_CONSTANT)

#        image = border
#        x_start = 0
#        y_start = 0
#        for nx in range(int(n_crops_x)):
#                # print(nx)
#                for ny in range(int(n_crops_y)):
#                        crop = image[x_start:x_start+size_x, y_start:y_start+size_y].copy()
#                        crops.append(crop)
#                        # print(crop.shape, x_start, y_start)
#                        y_start += size_y
#                y_start = 0
#                x_start += size_x

#        return crops

#img = cv2.imread("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_rgb_img/rgb_00023.png")
#crop_img = crop_image(img, (448, 448))

#img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_rgb_img/rgb_00023.png")
#img = tf.image.decode_png(img, 3)

#n_crops_x = tf.math.ceil(img.shape[0] / float(448)) 
#n_crops_y = tf.math.ceil(img.shape[1] / float(448)) 
#required_x = n_crops_x * 448
#required_y = n_crops_y * 448
#required_x = required_x - img.shape[0]
#required_y = required_y - img.shape[1]
#required_x /= 2.
#required_y /= 2.

#border = tf.pad(img, [[required_x, required_x],[required_y, required_y],[0,0]])

#h = tf.random.uniform([1], 0, border.shape[0] - 448)
#h = tf.cast(tf.math.ceil(h[0]), tf.int32)
#w = tf.random.uniform([1], 0, border.shape[1] - 448)
#w = tf.cast(tf.math.ceil(w[0]), tf.int32)

#i = border[h:h+448, w:w+448, :]


#a = 0

