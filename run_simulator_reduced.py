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
            f'data/simulator/reduced/f{data_string}.rawmax200.evmax30.b128.ed1.test.dataset'
        )
    else:
        dm.setup()
        train_ds = dm.dataset_train
        dm.dir = 'data/simulator/reduced/{}.eval'.format(data_string)
        dm.train_size, dm.val_size, dm.test_size = 0, 0.25, 0.75
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

    batch_loss = utils.BatchLogs('batch_loss')

    # Callbacks
    checkpoint_filepath = f'models/simulator.reduced/model.{name}/model_chp'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_batch_loss',
        mode='min',
        save_best_only=True
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_batch_loss',
        patience=patience,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'info/simulator.reduced/csvlog.{name}.log',
    )
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()

    start = timer()
    hist = basecaller.fit(
        train_ds,
        epochs=epochs,
        callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback, csv_logger, nan_terminate],
        validation_data=val_ds
    )
    mid_1 = timer()

    # load best model by val_batch_loss
    basecaller.load_weights(checkpoint_filepath)
    print(name_spec)
    print(hist.history)

    info = {}
    info['train_history'] = hist.history

    mid_2 = timer()
    test_accuracy = basecaller.evaluate_test(test_ds)
    end = timer()
    print('Test accuracy: {}'.format(test_accuracy))

    info['test_accuracy'] = test_accuracy

    info['train_time'] = mid_1 - start
    info['test_time'] = end - mid_2

    info['batch_loss'] = batch_loss.logs

    with open(f'info/simulator.reduced/info.{name}.json', 'w') as info_file:
        json.dump(info, info_file, indent=2)

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
        'bases_offset': 1,
        'rnn_type': '',
        'attention_type': 'bahdanau',
        'embedding_dim': 5,
        'event_detection': True
    }

    variants = [
        {'data_string': 'seq.3.25000.45', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.3.25000.45', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.3.25000.45', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.12.75000.450', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.12.75000.450', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.12.75000.450', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.21.150000.1024', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.21.150000.1024', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.21.150000.1024', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.43.300000.2048', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.43.300000.2048', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.43.300000.2048', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.4096.600000.4096', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.4096.600000.4096', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        {'data_string': 'seq.4096.600000.4096', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
    ]

    for v in variants:
        run_simulator_training(
            **{**basic_params, **v}
        )