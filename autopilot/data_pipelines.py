import os

import dicto as do

import numpy as np
import pandas as pd
import tensorflow as tf

CSV_FILENAME = "driving_log.csv"

def serving_input_fn(params):

    input_image = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, None, 3],
        name="input_image"
    )

    images = tf.image.resize_images(input_image, [params.image_height, params.image_width])


    crop_window = get_crop_window(params)
    images = tf.image.crop_to_bounding_box(images, *crop_window)

    images = tf.image.resize_images(images, [params.resize_height, params.resize_width])

    images = (images / 255.0) * 2.0 - 1.0

    return tf.estimator.export.ServingInputReceiver(
        features = dict(
            image=images,
        ),
        receiver_tensors = dict(
            image=input_image,
        ),
    )