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

def input_fn(data_dir, params):

    csv_filepath = os.path.join(data_dir, CSV_FILENAME)
    df = pd.read_csv(csv_filepath)
    df = normalize_dataframe(df)

    # There are whitespaces problems for column filename
    # df["filepath"] = df["filename"].str.strip()
    df["filepath"] = df["filename"].apply(lambda row: os.path.join(data_dir, row.strip()))

    df = process_dataframe(df, params)

    if params.only_center_camera:
        df = df[df.camera == 1]


    # Shuffle the dataset
    df = df.sample(frac=1)

    tensors = dict(
        filepath=df.filepath.values,
        steering=df.steering.values,
        camera=df.camera.values,
        original_steering=df.original_steering.values,
    )

    if "flipped" in df:
        tensors["flipped"] = df.flipped.values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(tensors)

    ds = ds.shuffle(buffer_size=params.buffer_size, reshuffle_each_iteration=True)

    ds = ds.apply(tf.data.experimental.map_and_batch(
        lambda row: process_data(row, params),
        batch_size=params.batch_size,
        num_parallel_calls=params.n_threads,
        drop_remainder=True,
    ))

    # Please check dataset options
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)

    return ds