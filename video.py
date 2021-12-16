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
from keras.preprocessing.image import load_img


if __name__ == "__main__":
    # cap = cv2.VideoCapture('10s_sample.mp4')

    # if cap.isOpened():
    #     cnt = 0
    #     while 1:
    #         cnt += 1
    #         ret, frame = cap.read()
    #         if ret and cnt % 20 == 0:
    #             cv2.imshow('frame', frame)
    #         k = cv2.waitKey(10)
    #         if k == ord('q'):
    #             break
    #     cv2.destroyAllWindows()

    model = load_model('deeplabv3plus_retry_cs_kitti_xception.hdf5', custom_objects={
        "categorical_crossentropy_plus_jaccard_loss": sm.losses.cce_jaccard_loss, "iou_score": sm.metrics.iou_score})

    classes = pd.read_csv('data/classes.csv', index_col='Unnamed: 0')
    classes = classes[4:]
    cls2rgb = {cl: list(classes.loc[cl, :]) for cl in classes.index}
    idx2rgb = {idx: np.array(rgb)
               for idx, (cl, rgb) in enumerate(cls2rgb.items())}

    def map_class_to_rgb(p):
        return idx2rgb[p[0]]

    cap = cv2.VideoCapture('cairo_4k_10s_sample.mp4')

    # out = cv2.VideoWriter(
    #     'outpy.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (256, 256))

    # Loop until the end of the video
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (256, 256))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('video', frame)

        x = frame
        x = x/255.
        x = np.expand_dims(x, 0)
        pred = model.predict(x)
        # rgb_mask = np.apply_along_axis(
        #     map_class_to_rgb, -1, np.expand_dims(pred[0], -1))
        rgb_mask = np.apply_along_axis(
            map_class_to_rgb, -1, np.expand_dims(np.argmax(pred[0], axis=-1), -1))
        rgb_mask = rgb_mask.astype('uint8')
        rgb_mask = cv2.resize(rgb_mask, (256, 256))

        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', rgb_mask)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
