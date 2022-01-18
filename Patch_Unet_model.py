# -*- coding:utf-8 -*-

import tensorflow as tf


def Path_Unet(input_shape=(512, 512, 3), classes=3):
    
    h = inputs = tf.keras.Input(input_shape)

    h1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h)
    h1 = tf.keras.layers.Dropout(0.1)(h1)
    h1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(h1)

    h2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(p1)
    h2 = tf.keras.layers.Dropout(0.1)(h2)
    h2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h2)
    p2 = tf.keras.layers.MaxPool2D((2,2))(h2)

    h3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(p2)
    h3 = tf.keras.layers.Dropout(0.2)(h3)
    h3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h3)
    p3 = tf.keras.layers.MaxPool2D((2,2))(h3)

    h4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(p3)
    h4 = tf.keras.layers.Dropout(0.2)(h4)
    h4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h4)
    p4 = tf.keras.layers.MaxPool2D((2,2))(h4)

    h5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(p4)
    h5 = tf.keras.layers.Dropout(0.3)(h5)
    h5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h5)

    d6 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="same")(h5)
    d6 = tf.keras.layers.concatenate([d6, h4], -1)
    h6 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(d6)
    h6 = tf.keras.layers.Dropout(0.2)(h6)
    h6 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h6)

    d7 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="same")(h6)
    d7 = tf.keras.layers.concatenate([d7, h3], -1)
    h7 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(d7)
    h7 = tf.keras.layers.Dropout(0.2)(h7)
    h7 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h7)

    d8 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="same")(h7)
    d8 = tf.keras.layers.concatenate([d8, h2], -1)
    h8 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(d8)
    h8 = tf.keras.layers.Dropout(0.1)(h8)
    h8 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(h8)

    d9 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, padding="same")(h8)
    d9 = tf.keras.layers.concatenate([d9, h1], -1)
    h9 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='elu', kernel_initializer="he_normal", padding="same")(d9)
    h9 = tf.keras.layers.Dropout(0.1)(h9)

    h10 = tf.keras.layers.Conv2D(filters=classes, kernel_size=1, strides=1)(h9)

    return tf.keras.Model(inputs=inputs, outputs=h10)


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

