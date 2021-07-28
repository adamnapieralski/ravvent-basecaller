import tensorflow as tf

def masked_accuracy(y_true, y_pred, omit_vals):
    match = tf.cast(y_true == y_pred, tf.int64)

    mask = tf.ones_like(y_true)
    for ov in omit_vals:
        mask *= tf.cast(y_true != ov, tf.int64)

    total = tf.reduce_sum(mask)
    count = tf.reduce_sum(mask * match)
    return count / total

def input_mask(input_sequence, padding_value):
    '''Get mask on input data that is not padding
        Parameters:
            input_sequence (tf.Tensor): shape = (batch_sz, time, features)
            padding_value (int): padded elements consist only of padding_value (e.g. [-1], [-1, -1, -1])
    '''
    return tf.reduce_all(input_sequence == padding_value, axis=-1)

def unpack_data_to_input_target(data, input_data_type):
    raw_sequence, events_sequence, target_sequence = data

    if input_data_type == 'raw':
        input_sequence = raw_sequence
    elif input_data_type == 'event':
        input_sequence = events_sequence
    return input_sequence, target_sequence