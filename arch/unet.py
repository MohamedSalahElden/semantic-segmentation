import tensorflow as tf


def Unet(DIM, no_of_classes):
    inputs = tf.keras.layers.Input((DIM, DIM, 3))
    s = inputs

    # 1st block
    c1 = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)

    # c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    # 2nd bloc
    c2 = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    # c2 = tf.keras.layers.Dropout(0.1)(c2)

    c2 = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # 3rd block
    c3 = tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    # c3 = tf.keras.layers.Dropout(0.1)(c3)

    c3 = tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # 4th block

    c4 = tf.keras.layers.Conv2D(
        256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.1)(c4)

    c4 = tf.keras.layers.Conv2D(
        256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # u-base block

    c5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    # c5 = tf.keras.layers.Dropout(0.3)(c5)

    c5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # expansion bloc (decoder)

    # 6th block
    u6 = tf.keras.layers.Conv2DTranspose(
        256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(256, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(256, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c6)
    # 7th block

    u7 = tf.keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(128, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(128, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c7)

    # 8th block
    u8 = tf.keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(64, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(u8)
    # c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(64, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c8)

    # 9th block
    u9 = tf.keras.layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(32, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(no_of_classes, (3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c9)

    # outputs
    outputs = tf.keras.layers.Conv2D(
        no_of_classes, (1, 1), activation='softmax')(c9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model

    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=jaccard_distance, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=31)])
    # # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=31)])

    # model.summary()
