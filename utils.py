import tensorflow as tf

def masked_accuracy(y_true, y_pred, omit_vals):
    match = tf.cast(y_true == y_pred, tf.int64)

    mask = tf.ones_like(y_true)
    for ov in omit_vals:
        mask *= tf.cast(y_true != ov, tf.int64)

    total = tf.reduce_sum(mask)
    count = tf.reduce_sum(mask * match)
    return count / total