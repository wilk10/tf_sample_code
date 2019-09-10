import tensorflow as tf


class Model:

    @staticmethod
    def declare(input_shape, base_filters=16):
        f = [base_filters * (2 ** i) for i in range(5)]
        inputs = tf.keras.layers.Input(input_shape)
        c1 = tf.keras.layers.Conv2D(f[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Conv2D(f[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(f[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(f[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(f[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(f[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(f[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Conv2D(f[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        d4 = tf.keras.layers.Dropout(0.5)(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(d4)

        c5 = tf.keras.layers.Conv2D(f[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Conv2D(f[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        d5 = tf.keras.layers.Dropout(0.5)(c5)

        u6 = tf.keras.layers.UpSampling2D((2, 2))(d5)
        c6 = tf.keras.layers.Conv2D(f[3], (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        m6 = tf.keras.layers.concatenate([d4, c6], axis=3)
        c6 = tf.keras.layers.Conv2D(f[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m6)
        c6 = tf.keras.layers.Conv2D(f[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.UpSampling2D((2, 2))(c6)
        c7 = tf.keras.layers.Conv2D(f[2], (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        m7 = tf.keras.layers.concatenate([c3, c7], axis=3)
        c7 = tf.keras.layers.Conv2D(f[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m7)
        c7 = tf.keras.layers.Conv2D(f[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.UpSampling2D((2, 2))(c7)
        c8 = tf.keras.layers.Conv2D(f[1], (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        m8 = tf.keras.layers.concatenate([c2, c8], axis=3)
        c8 = tf.keras.layers.Conv2D(f[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m8)
        c8 = tf.keras.layers.Conv2D(f[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.UpSampling2D((2, 2))(c8)
        c9 = tf.keras.layers.Conv2D(f[0], (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        m9 = tf.keras.layers.concatenate([c1, c9], axis=3)
        c9 = tf.keras.layers.Conv2D(f[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(m9)
        c9 = tf.keras.layers.Conv2D(f[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = tf.keras.layers.Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        return tf.keras.Model(inputs=[inputs], outputs=[outputs])

    @classmethod
    def keras_estimator(cls, model_dir, input_shape, base_filters):
        model = cls.declare(input_shape, base_filters)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)
