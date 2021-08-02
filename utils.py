import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle

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
        input_data = raw_sequence
    elif input_data_type == 'event':
        input_data = events_sequence
    elif input_data_type == 'joint':
        input_data = (raw_sequence, events_sequence)
    return input_data, target_sequence

def train_val_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1, random_state=None, shuffle=True, stratify=None):
    if train_size + val_size + test_size != 1:
        raise Exception('Train/validation/test datasets fractions don\'t sum up to 1.')

    data_sh = data
    if shuffle:
        data_sh = sklearn_shuffle(data, random_state=random_state)

    if train_size > 0 and train_size < 1:
        train, test = train_test_split(data_sh, test_size=test_size+val_size, random_state=random_state, shuffle=False, stratify=stratify)
        if test_size == 0:
            return train, test, None
        if val_size == 0:
            return train, None, test
        val, test = train_test_split(test, train_size=(val_size / (val_size + test_size)), random_state=random_state, shuffle=False, stratify=stratify)
        return train, val, test
    if train_size == 0:
        if val_size == 1:
            return None, data_sh, None
        if test_size == 1:
            return None, None, data_sh
        val, test = train_test_split(data_sh, test_size=test_size, random_state=random_state, shuffle=False, stratify=stratify)
        return None, val, test
    if train_size == 1:
        return data_sh, None, None
