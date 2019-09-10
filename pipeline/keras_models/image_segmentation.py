import numpy as np
from tensorflow import keras


class Rudimental(keras.Model):

    @staticmethod
    def declare(input_shape):
        return keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation="relu", padding='same', input_shape=input_shape),
            keras.layers.Conv2D(16, (3, 3), activation="relu", padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(16, (3, 3), activation="relu", padding='same'),
            keras.layers.UpSampling2D((2, 2)),
            keras.layers.Conv2D(16, (3, 3), activation="relu", padding='same'),
            keras.layers.UpSampling2D((2, 2)),
            keras.layers.Conv2D(8, (3, 3), activation="relu", padding='same'),
            keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        ])


class PVclassifier(keras.Model):

    @staticmethod
    def add_relative_luminescences(images):
        coefficients = [0.2126, 0.7152, 0.0722]
        concatenated_images = []
        for image in images:
            products = []
            for i, coefficient in enumerate(coefficients):
                product = coefficient * image[:, :, i]
                products.append(product)
            relative_luminescence = np.sum(np.array(products), axis=0)
            relative_luminescence = np.expand_dims(relative_luminescence, axis=2)
            concatenated_image = np.concatenate((image, relative_luminescence), axis=2)
            concatenated_images.append(concatenated_image)
        return np.array(concatenated_images)

    @staticmethod
    def declare(input_shape):
        inputs = keras.layers.Input(input_shape)
        c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = keras.layers.Dropout(0.1)(c1)
        c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = keras.layers.Dropout(0.2)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = keras.layers.Dropout(0.3)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)

        u4 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
        u4 = keras.layers.concatenate([u4, c2])
        c4 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u4)
        c4 = keras.layers.Dropout(0.2)(c4)
        c4 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)

        u5 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = keras.layers.concatenate([u5, c1])
        c5 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u5)
        c5 = keras.layers.Dropout(0.1)(c5)
        c5 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
        return keras.Model(inputs=[inputs], outputs=[outputs])


class SimplifiedUnet:

    @staticmethod
    def declare(input_shape):
        inputs = keras.layers.Input(input_shape)
        c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = keras.layers.Dropout(0.1)(c1)
        c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = keras.layers.Dropout(0.2)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = keras.layers.Dropout(0.2)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = keras.layers.Dropout(0.3)(c4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)

        u5 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = keras.layers.concatenate([u5, c3])
        c5 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u5)
        c5 = keras.layers.Dropout(0.2)(c5)
        c5 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = keras.layers.concatenate([u6, c2])
        c6 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = keras.layers.Dropout(0.2)(c6)
        c6 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c1])
        c7 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = keras.layers.Dropout(0.1)(c7)
        c7 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
        return keras.Model(inputs=[inputs], outputs=[outputs])


class FullUnet:
    # based on: https://github.com/zhixuhao/unet/blob/master/model.py

    @staticmethod
    def declare(input_shape):
        inputs = keras.layers.Input(input_shape)
        c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        d4 = keras.layers.Dropout(0.5)(c4)
        p4 = keras.layers.MaxPooling2D((2, 2))(d4)

        c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        d5 = keras.layers.Dropout(0.5)(c5)

        u6 = keras.layers.UpSampling2D((2, 2))(d5)
        c6 = keras.layers.Conv2D(512, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        m6 = keras.layers.concatenate([d4, c6], axis=3)
        c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m6)
        c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = keras.layers.UpSampling2D((2, 2))(c6)
        c7 = keras.layers.Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        m7 = keras.layers.concatenate([c3, c7], axis=3)
        c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m7)
        c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = keras.layers.UpSampling2D((2, 2))(c7)
        c8 = keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        m8 = keras.layers.concatenate([c2, c8], axis=3)
        c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m8)
        c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = keras.layers.UpSampling2D((2, 2))(c8)
        c9 = keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        m9 = keras.layers.concatenate([c1, c9], axis=3)
        c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m9)
        c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = keras.layers.Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        return keras.Model(inputs=[inputs], outputs=[outputs])
