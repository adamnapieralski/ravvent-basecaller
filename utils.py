import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
import subprocess
import shlex

from pathlib import Path

from shape_checker import ShapeChecker

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
    return tf.reduce_all(input_sequence != padding_value, axis=-1)

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

def get_bases_sequence_from_chiron_dir(dir: str, max_length: int = None):
    dir = Path(dir)
    labels_paths = [p for p in dir.iterdir() if p.suffix == '.label']
    labels_paths.sort()

    bases_sequence = ''

    for label_path in labels_paths:
        labels = np.loadtxt(label_path, dtype='object')
        single_seq = labels[:,2]
        bases_sequence += ''.join(single_seq.tolist())
        if max_length is not None and len(bases_sequence) >= max_length:
            bases_sequence = bases_sequence[0:max_length]
            break

    return bases_sequence

def create_fastq_file(path, id, sequence):
    with open(path, 'w') as f:
        f.write(f'${id}')
        f.write(sequence)

def run_minimap(reference_fasta, basecalled_fastq, output_alignment):
    cmd = f'minimap2 {reference_fasta} {basecalled_fastq} > {output_alignment}'
    subprocess.run(shlex.split(cmd))

def get_accuracy_alignment_from_minimap(output_alignment):
    stats = np.loadtxt(output_alignment, dtype='object', skiprows=1)
    return int(stats[10]) / int(stats[11])
class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, padding_value=0):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.padding_value = padding_value

    def __call__(self, y_true, y_pred):
        shape_checker = ShapeChecker()
        shape_checker(y_true, ('batch', 't'))
        shape_checker(y_pred, ('batch', 't', 'logits'))

        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)
        shape_checker(loss, ('batch', 't'))

        # Mask off the losses on padding.
        mask = tf.cast(y_true != self.padding_value, tf.float32)
        shape_checker(mask, ('batch', 't'))
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)
