import tensorflow as tf
from timeit import default_timer as timer
import json

from data_loader import DataGenerator, DataModule
from basecaller import Basecaller
import utils

RANDOM_SEED = 22

def run_chiron_training(
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

    load_source = 'chiron'
    teacher_forcing = False

    if data_type == 'joint':
        name_max_len = f'rawmax{raw_max_len}.evmax{event_max_len}'
    else:
        name_max_len = f'rawmax{raw_max_len}' if data_type == 'raw' else f'evmax{event_max_len}'

    name_spec = f'{data_type}.{rnn_type}.u{units}.{load_source}.{name_max_len}.b{batch_size}.ep{epochs}.pat{patience}.tf{int(teacher_forcing)}.emb{embedding_dim}.ed{int(event_detection)}.{attention_type}'

    name = name_spec
    print('RUNNING', name)
    dm = DataModule(
        dir='',
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
    dm.load_scalers('data/chiron/lambda/train/all/scalers.rawmax200.evmax30.offset6.pkl')
    dm.calculate_input_padding_value()


    dg_train = DataGenerator('data/chiron/lambda/train/all/samples.rawmax200.evmax30.offset6/files_info.json', batch_size=batch_size, shuffle=True)
    dg_val = DataGenerator('data/chiron/lambda/eval/all/samples.rawmax200.evmax30.offset6/files_info_val.json', batch_size=batch_size, shuffle=False)

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

    batch_loss = utils.BatchLogs('loss')

    # Callbacks
    checkpoint_filepath = f'models/chiron/model.{name}.{"{epoch:02d}"}/model_chp'
    print(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'info/chiron/csvlog.{name}.log', append=False
    )
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()

    start = timer()
    hist = basecaller.fit(
        dg_train,
        epochs=epochs,
        callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback, csv_logger, nan_terminate],
        validation_data=dg_val
    )
    mid_1 = timer()

    # # load best model by val_batch_loss
    # basecaller.load_weights(checkpoint_filepath)
    # print(name_spec)
    # print(hist.history)

    info = {}
    info['train_history'] = hist.history

    # mid_2 = timer()
    # test_accuracy = basecaller.evaluate_test(test_ds)
    # end = timer()
    # print('Test accuracy: {}'.format(test_accuracy))

    # info['test_accuracy'] = test_accuracy

    info['train_time'] = mid_1 - start
    # info['test_time'] = end - mid_2

    info['batch_loss'] = batch_loss.logs

    with open(f'info/chiron/info.{name}.json', 'w') as info_file:
        json.dump(info, info_file, indent=2)

if __name__ == '__main__':
    basic_params = {
        'data_type': '',
        'batch_size': 128,
        'raw_max_len': 200,
        'event_max_len': 30,
        'units': 128,
        'epochs': 30,
        'patience': 30,
        'bases_offset': 6,
        'rnn_type': '',
        'attention_type': 'bahdanau',
        'embedding_dim': 5,
        'event_detection': True
    }

    variants = [
        {'data_type': 'raw', 'rnn_type': 'bilstm'},
    ]

    for v in variants:
        run_chiron_training(
            **{**basic_params, **v}
        )