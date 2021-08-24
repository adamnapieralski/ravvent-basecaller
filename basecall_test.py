import tensorflow as tf
from timeit import default_timer as timer
import json

from data_loader import DataModule
from basecaller import Basecaller
import utils

# DATA_TYPE = 'raw'

# BATCH_SIZE = 128

# RAW_MAX_LEN = 200
# EVENT_MAX_LEN = 30

# UNITS = 128
# EPOCHS = 100
# PATIENCE = 15
# DATA_PATH = 'data/chiron/train'
# BASES_OFFSET = 1
# TEACHER_FORCING = False
# RNN_TYPE = 'gru'
# LOAD_SOURCE = 'simulator'
# ATTENTION_TYPE = 'bahdanau' # 'luong'
# EMBEDDING_DIM = 5

RANDOM_SEED = 22

# if DATA_TYPE == 'joint':
#     NAME_MAX_LEN = f'rawmax{RAW_MAX_LEN}.evmax{EVENT_MAX_LEN}'
# else:
#     NAME_MAX_LEN = f'rawmax{RAW_MAX_LEN}' if DATA_TYPE == 'raw' else f'evmax{EVENT_MAX_LEN}'

# NAME_SPEC = f'{DATA_TYPE}.{RNN_TYPE}.u{UNITS}.{LOAD_SOURCE}.{NAME_MAX_LEN}.b{BATCH_SIZE}.ep{EPOCHS}.pat{PATIENCE}.tf{int(TEACHER_FORCING)}.emb{EMBEDDING_DIM}.{ATTENTION_TYPE}'

def run_simulator_training(
    data_string: str,
    data_type: str,
    batch_size: int,
    raw_max_len: int,
    event_max_len: int,
    units: int,
    epochs: int,
    patience: int,
    bases_offset: int,
    rnn_type: str,
    attention_type: str,
    embedding_dim: int,
    event_detection: bool
):
    tf.random.set_seed(RANDOM_SEED)

    load_source = 'simulator'
    teacher_forcing = False

    if data_type == 'joint':
        name_max_len = f'rawmax{raw_max_len}.evmax{event_max_len}'
    else:
        name_max_len = f'rawmax{raw_max_len}' if data_type == 'raw' else f'evmax{event_max_len}'

    name_spec = f'{data_type}.{rnn_type}.u{units}.{load_source}.{name_max_len}.b{batch_size}.ep{epochs}.pat{patience}.tf{int(teacher_forcing)}.emb{embedding_dim}.ed{int(event_detection)}.{attention_type}'

    name = name_spec + '.reduced.' + data_string
    print('RUNNING', name)
    dm = DataModule(
        dir='data/simulator/reduced/{}.train'.format(data_string),
        max_raw_length=raw_max_len,
        max_event_length=event_max_len,
        bases_offset=bases_offset,
        batch_size=batch_size,
        train_size=1,
        val_size=0,
        test_size=0,
        load_source=load_source,
        event_detection=event_detection,
        random_seed=RANDOM_SEED,
        shuffle=False,
        verbose=True
    )
    if data_string == 'seq.4096.600000.4096':
        train_ds = tf.data.experimental.load(
            f'data/simulator/reduced/{data_string}.rawmax200.evmax30.b128.ed1.train.dataset'
        )
        val_ds = tf.data.experimental.load(
            f'data/simulator/reduced/{data_string}.rawmax200.evmax30.b128.ed1.val.dataset'
        )
        test_ds = tf.data.experimental.load(
            f'data/simulator/reduced/{data_string}.rawmax200.evmax30.b128.ed1.test.dataset'
        )
    else:
        dm.bases_offset = 1
        dm.setup()
        train_ds = dm.dataset_train
        dm.dir = 'data/simulator/reduced/{}.eval'.format(data_string)
        dm.train_size, dm.val_size, dm.test_size = 0, 0, 1
        dm.bases_offset = 8
        dm.setup()
        val_ds, test_ds = dm.dataset_val, dm.dataset_test

    basecaller = Basecaller(
        units=units,
        output_text_processor=dm.output_text_processor,
        input_data_type=data_type,
        input_padding_value=dm.input_padding_value,
        embedding_dim=embedding_dim,
        rnn_type=rnn_type,
        teacher_forcing=teacher_forcing,
        attention_type=attention_type
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(),
        loss=utils.MaskedLoss(basecaller.output_padding_token),
    )

    name = name.replace('10000', '25000')
    checkpoint_filepath = f'models/simulator.reduced/model.{name}/model_chp'
    basecaller.load_weights(checkpoint_filepath)

    bases_sequences = basecaller.basecall_full(test_ds)
    print(bases_sequences)


if __name__ == '__main__':
    basic_params = {
        'data_string': '',
        'data_type': '',
        'batch_size': 128,
        'raw_max_len': 200,
        'event_max_len': 30,
        'units': 128,
        'epochs': 100,
        'patience': 100,
        'bases_offset': 5,
        'rnn_type': '',
        'attention_type': 'bahdanau',
        'embedding_dim': 5,
        'event_detection': True
    }

    variants = [
        {'data_string': 'seq.3.10000.45', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.3.25000.45', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.3.25000.45', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.12.75000.450', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.12.75000.450', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.12.75000.450', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.21.150000.1024', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.21.150000.1024', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.21.150000.1024', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.43.300000.2048', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.43.300000.2048', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.43.300000.2048', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.4096.600000.4096', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.4096.600000.4096', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_string': 'seq.4096.600000.4096', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
    ]

    for v in variants:
        run_simulator_training(
            **{**basic_params, **v}
        )