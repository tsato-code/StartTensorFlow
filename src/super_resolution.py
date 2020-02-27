import os
import glob
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Add, Conv2D, Conv2DTranspose, Dense, Input, MaxPooling2D, UpSampling2D, Lambda
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.utils import plot_model


DATA_DIR = "data/samples/chap10/data/chap10"
N_TRAIN_DATA = 1000
N_TEST_DATA = 100
BATCH_SIZE = 32


# データ前処理
def drop_resolution(x, scale=3.0):
    size = (x.shape[0], x.shape[1])
    small_size = (int(size[0]/scale), int(size[1]/scale))
    img = array_to_img(x)
    small_img = img.resize(small_size, 3)
    return img_to_array(small_img.resize(img.size, 3))


def data_generator(data_dir, mode, scale=2.0, target_size=(200, 200), batch_size=32, shuffle=True):
    for imgs in ImageDataGenerator().flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle
    ):
        x = np.array([drop_resolution(img, scale) for img in imgs])
        yield x/255., imgs/255.


def psnr(y_true, y_pred):
    return -10 * K.log(K.mean(K.flatten((y_true - y_pred))**2)) / np.log(10)


def main():
    train_data_generator = data_generator(DATA_DIR, "train", batch_size=BATCH_SIZE)
    test_x, test_y = next(data_generator(DATA_DIR, "test", batch_size=N_TEST_DATA, shuffle=False))

    model = Sequential()
    model.add(Conv2D(
        filters=64,
        kernel_size=9,
        padding="same",
        activation="relu",
        input_shape=(None, None, 3)
    ))

    model.add(Conv2D(
        filters=32,
        kernel_size=1,
        padding="same",
        activation="relu"
    ))

    model.add(Conv2D(
        filters=3,
        kernel_size=5,
        padding="same"
    ))

    model.summary()
    plot_model(
        model,
        to_file="./01_super_resolution.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True)

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[psnr]
    )

    model.fit_generator(
        train_data_generator,
        validation_data=(test_x, test_y),
        steps_per_epoch=N_TRAIN_DATA//BATCH_SIZE,
        epochs=2
    )

    pred = model.predict(test_x)

    inputs = Input((None, None, 3), dtype="float")
    conv1 = Conv2D(64, 3, padding="same")(inputs)
    conv1 = Conv2D(64, 3, padding="same")(conv1)

    conv2 = Conv2D(64, 3, strides=2, padding="same")(conv1)
    conv2 = Conv2D(64, 3, padding="same")(conv2)

    conv3 = Conv2D(64, 3, strides=2, padding="same")(conv2)
    conv3 = Conv2D(64, 3, padding="same")(conv3)

    deconv3 = Conv2DTranspose(64, 3, padding="same")(conv3)
    deconv3 = Conv2DTranspose(64, 3, strides=2, padding="same")(deconv3)

    merge2 = Add()([deconv3, conv2])
    deconv2 = Conv2DTranspose(64, 3, padding="same")(merge2)
    deconv2 = Conv2DTranspose(64, 3, strides=2, padding="same")(deconv2)

    merge1 = Add()([deconv2, conv1])
    deconv1 = Conv2DTranspose(64, 3, padding="same")(merge1)
    deconv1 = Conv2DTranspose(3, 3, strides=2, padding="same")(deconv1)

    output = Add()([deconv1, inputs])

    model = Model(inputs, output)

    model.summary()
    plot_model(
        model,
        to_file="./02_super_resolution.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True)


if __name__ == "__main__":
    main()
