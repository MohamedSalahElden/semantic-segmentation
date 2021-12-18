from arch import unet
import tensorflow as tf
from keras.models import load_model
from utils.util import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import segmentation_models as sm
from utils.labels import map_class_to_rgb


if __name__ == "__main__":

    base_data_dir = 'data'
    saved_models = 'checkpoint'

    train_path = os.path.join(base_data_dir, 'data')
    test_path = os.path.join(base_data_dir, 'testing')

    classes_path = os.path.join(base_data_dir, 'classes.csv')

    iti_data_dir = os.path.join(base_data_dir, 'aroundITI')
    single_frames_dir = os.path.join(base_data_dir, 'single_frames')
    videos_dir = os.path.join(base_data_dir, 'videos')

    output_images_dir = os.path.join(base_data_dir, 'output_images')
    output_videos_dir = os.path.join(base_data_dir, 'output_videos')


# best so far deeplabv3plus_retry_cs_kitti_xception.hdf5

    model_name = "deeplabv3plus_retry_cs_kitti_xception.hdf5"
    DIM = 256
    used_model = os.path.join(saved_models, model_name)

    model = load_model(used_model, custom_objects={
        "categorical_crossentropy_plus_jaccard_loss": sm.losses.cce_jaccard_loss, "iou_score": sm.metrics.iou_score})

    classes = pd.read_csv(classes_path, index_col='name')

    sample_video = os.path.join(videos_dir, 'cairo 4k_sample2.mp4')
    cap = cv2.VideoCapture(sample_video)

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (DIM, DIM))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('video', frame)

        x = frame
        x = x/255.
        x = np.expand_dims(x, 0)
        pred = model.predict(x)

        rgb_mask = np.apply_along_axis(
            map_class_to_rgb, -1, np.expand_dims(np.argmax(pred[0], axis=-1), -1))
        rgb_mask = rgb_mask.astype('uint8')
        rgb_mask = cv2.resize(rgb_mask, (DIM, DIM))

        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', rgb_mask)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
