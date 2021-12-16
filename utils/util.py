import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

import cv2
from utils.labels import *


##########
IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 35
##########


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())


def loss(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy(y_true, y_pred) - tf.math.log(dice_coefficient(y_true, y_pred) + tf.keras.backend.epsilon())


def visualize_seg(idx2rgb, model, img, gt_mask, shape='normal', gt_mode='sparse'):
    #     plt.figure(1)

    #   # Img
    #     plt.subplot(311)
    #     plt.imshow(img)

    # Predict
    pred_mask = model.predict(np.expand_dims(img, 0))
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]
    if shape == 'flat':
        # Reshape only if you use the flat model. O.w. you dont need
        pred_mask = np.reshape(pred_mask, (128, 128))

    rgb_mask = np.apply_along_axis(
        map_class_to_rgb, -1, np.expand_dims(pred_mask, -1), idx2rgb)

    # Prediction
    # plt.subplot(312)
    # plt.imshow(rgb_mask)

    # GT mask
    if gt_mode == 'ohe':
        gt_img_ohe = np.argmax(gt_mask, axis=-1)
        gt_mask = np.apply_along_axis(
            map_class_to_rgb, -1, np.expand_dims(gt_img_ohe, -1))

    # plt.subplot(313)
    # plt.imshow((gt_mask).astype(np.uint8))

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(rgb_mask)
    ax[2].imshow((gt_mask).astype(np.uint8))
    plt.show()


def read_data(train_path, images_path, masks_path):
    image_path = os.path.join(train_path, images_path)
    mask_path = os.path.join(train_path, masks_path)
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    image_list = sorted([image_path+i for i in image_list])
    mask_list = sorted([mask_path+i for i in mask_list])

    return image_list, mask_list


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    image_list = tf.constant(image_list)
    mask_list = tf.constant(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def read_image(image_path):
    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 127.5 - 1
    return image


def load_data(image_list):
    image = read_image(image_list)
    return image


def data_generator(image_list):
    image_list = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def predict_images(model, img_list, DIM, output_width, output_height, directory):
    i = 0
    for file in img_list:
        i += 1
        x = np.array(
            [np.array(load_img(str(directory)+'/' + file, target_size=(DIM, DIM)))*1./255])
        pred = model.predict(x)

        write_output_images(pred, i, output_width, output_height)


def write_output_images(pred, i, output_width, output_height):
    rgb_mask = np.apply_along_axis(
        map_class_to_rgb, -1, np.expand_dims(np.argmax(pred[0], axis=-1), -1))
    rgb_mask = rgb_mask.astype('float32')
    rgb_mask = cv2.resize(rgb_mask, (output_width, output_height),)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)

    cv2.imwrite('res_{}.png'.format(i), rgb_mask)


def train_valid_generator(train_path, seed, batch_sz, DIM):
    data_gen_args = dict(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255)

    mask_gen_args = dict(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    #image_datagen.fit(images, augment=True, seed=seed)
    #mask_datagen.fit(masks, augment=True, seed=seed)
    t_path = os.path.join(train_path, 'train')
    train_length = len(os.listdir(os.path.join(t_path, 'images')))

    image_generator = image_datagen.flow_from_directory(
        t_path,
        class_mode=None,
        classes=['images'],
        seed=seed,
        batch_size=batch_sz,
        target_size=(DIM, DIM))

    mask_generator = mask_datagen.flow_from_directory(
        t_path,
        classes=['labels'],
        class_mode=None,
        seed=seed,
        batch_size=batch_sz,
        color_mode='rgb',
        target_size=(DIM, DIM))

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    def train_generator_fn():
        for (img, mask) in train_generator:
            new_mask = adjust_mask(mask)
            yield (img, new_mask)

    v_path = os.path.join(train_path, 'valid')
    valid_length = len(os.listdir(os.path.join(v_path, 'images')))

    val_image_generator = image_datagen.flow_from_directory(
        v_path,
        class_mode=None,
        classes=['images'],
        seed=seed,
        batch_size=batch_sz,
        target_size=(DIM, DIM))

    val_mask_generator = mask_datagen.flow_from_directory(
        v_path,
        classes=['labels'],
        class_mode=None,
        seed=seed,
        batch_size=batch_sz,
        color_mode='rgb',
        target_size=(DIM, DIM))

    # combine generators into one which yields image and masks
    val_generator = zip(val_image_generator, val_mask_generator)

    def val_generator_fn():

        for (img, mask) in val_generator:
            new_mask = adjust_mask(mask)
            yield (img, new_mask)

    return train_generator_fn, val_generator_fn, train_length, valid_length


def freeze_model_layers(model, layer_name):
    for layer in model.layers:
        # print(layer.name)
        if layer.name == layer_name:
            break
        else:
            layer.trainable = False


def plot_predictions(path, model, dim, mode=None, seed=None, test_sample_size=10):

    if mode == 'train' or mode == 'valid':
        img_path = os.path.join(path, mode, 'images')
        labels_path = os.path.join(path, mode, 'labels')

        for file, label in zip(sorted(os.listdir(img_path)), sorted(os.listdir(labels_path))):
            x = np.array(
                [np.array(load_img(os.path.join(img_path, file), target_size=(dim, dim)))*1./255])
            y = np.array(
                [np.array(load_img(os.path.join(labels_path, file), target_size=(dim, dim)))])
            preds = model.predict(x)

            fig, ax = plt.subplots(1, 4, figsize=(16, 16))
            rgb_mask = np.apply_along_axis(
                map_class_to_rgb, -1, np.expand_dims(np.argmax(preds[0], axis=-1), -1))

            ax[0].imshow(rgb_mask)
            ax[1].imshow(y[0])
            ax[2].imshow(x[0])

            ax[3].imshow(x[0])
            ax[3].imshow(rgb_mask, cmap='jet', alpha=0.5)

    elif mode == 'testing':
        np.random.seed(seed)
        testing_list = sorted(os.listdir(path))
        for file in np.random.choice(testing_list, size=test_sample_size, replace=False, ):
            x = np.array(
                [np.array(load_img(os.path.join(path, file), target_size=(dim, dim)))*1./255])
            preds = model.predict(x)
            fig, ax = plt.subplots(1, 3, figsize=(16, 16))
            rgb_mask = np.apply_along_axis(
                map_class_to_rgb, -1, np.expand_dims(np.argmax(preds[0], axis=-1), -1))
            ax[0].imshow(rgb_mask)
            ax[1].imshow(x[0])

            ax[2].imshow(x[0])
            ax[2].imshow(rgb_mask, cmap='jet', alpha=0.5)

    elif mode == 'ITI':
        iti_list = sorted(os.listdir(path))
        for file in iti_list:
            x = np.array(
                [np.array(load_img(os.path.join(path, file), target_size=(dim, dim)))*1./255])
            preds = model.predict(x)
            fig, ax = plt.subplots(1, 3, figsize=(16, 16))
            rgb_mask = np.apply_along_axis(
                map_class_to_rgb, -1, np.expand_dims(np.argmax(preds[0], axis=-1), -1))
            ax[0].imshow(rgb_mask)
            ax[1].imshow(x[0])

            ax[2].imshow(x[0])
            ax[2].imshow(rgb_mask, cmap='jet', alpha=0.5)
