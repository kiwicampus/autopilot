import tensorflow as tf
from tensorflow.python import autograph

from .model import pilot_relation_net

def get_weights(steering, params):


    ones = tf.ones_like(steering)
    zeros_weight = params.zeros_weight * ones

    weights = tf.where(
        tf.abs(steering) < 0.5,
        ones,
        ones * params.max_weight,
    )

    return tf.where(
        tf.equal(steering, 0.0),
        zeros_weight,
        weights,
    )


def get_onehot_labels(steering, params):
    
    label = tf.clip_by_value(steering, -1, 1)
    label = (label + 1.0) / 2.0
    label = label * (params.nbins - 1)

    label_upper = tf.ceil(label)
    label_lower = tf.floor(label)

    prob_upper = 1.0 - (label_upper - label)
    prob_upper = tf.cast(prob_upper, tf.float32)
    prob_upper = tf.expand_dims(prob_upper, 1)

    prob_lower = 1.0 - prob_upper

    onehot_upper = prob_upper * tf.one_hot(tf.cast(label_upper, tf.int32), params.nbins)
    onehot_lower = prob_lower * tf.one_hot(tf.cast(label_lower, tf.int32), params.nbins)

    onehot_labels = onehot_upper + onehot_lower

    return onehot_labels



@autograph.convert()
def get_learning_rate(params):
    
    global_step = tf.train.get_global_step()

    initial_learning_rate = params.learning_rate * params.batch_size / 128.0

    if global_step < params.cold_steps:
        learning_rate = params.cold_learning_rate
        learning_rate = tf.cast(learning_rate, tf.float32)
    
    elif global_step < params.cold_steps + params.warmup_steps:
        step = global_step - params.cold_steps
        p = step / params.warmup_steps
        
        learning_rate = initial_learning_rate * p + (1.0 - p) * params.cold_learning_rate
        learning_rate = tf.cast(learning_rate, tf.float32)

    else:
        step = global_step - (params.cold_steps + params.warmup_steps)
        learning_rate = tf.train.linear_cosine_decay(initial_learning_rate, step, params.decay_steps, beta = params.final_learning_rate)
        learning_rate = tf.cast(learning_rate, tf.float32)

    return learning_rate
