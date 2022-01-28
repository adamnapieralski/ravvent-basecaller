import tensorflow as tf
from timeit import default_timer as timer
import json

from data_loader import DataGenerator, DataModule
from basecaller_s2s import Basecaller
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

    load_source = 'simulator'
    teacher_forcing = 0.5
    learning_rate = 0.001

    if data_type == 'joint':
        name_max_len = f'rawmax{raw_max_len}.evmax{event_max_len}'
    else:
        name_max_len = f'rawmax{raw_max_len}' if data_type == 'raw' else f'evmax{event_max_len}'

    name_spec = f'{data_type}.lr{round(learning_rate, 6)}.{rnn_type}.u{units}.{load_source}.{name_max_len}.b{batch_size}.ep{epochs}.pat{patience}.emb{embedding_dim}.ed{int(event_detection)}.{attention_type}.tf{round(teacher_forcing, 2)}.do02.boff{bases_offset}'

    name = name_spec
    print('RUNNING', name)

    dm = DataModule(
        dir='data/simulator/reduced/seq.4096.600000.4096.train',
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
    dm.setup()
    train_ds = dm.dataset_train

    dm.dir = 'data/simulator/reduced/seq.4096.600000.4096.eval'
    dm.train_size, dm.val_size, dm.test_size = 0, 0.25, 0.75
    dm.setup()
    val_ds, test_ds = dm.dataset_val, dm.dataset_test

    # tf.data.experimental.save(
    #     train_ds, 'data/chiron/lambda/train_80_3'
    # )
    # tf.data.experimental.save(
    #     val_ds, 'data/chiron/lambda/val_80_3'
    # )
    # tf.data.experimental.save(
    #     val_ds, 'data/chiron/lambda/test_80_3'
    # )
    # train_ds = tf.data.experimental.load('data/chiron/lambda/train_80_3')
    # val_ds = tf.data.experimental.load('data/chiron/lambda/val_80_3')
    # train_ds = tf.data.experimental.load('data/chiron/lambda/train_80_3')

    # dg_train = DataGenerator('data/chiron/lambda/train/all/samples.rawmax200.evmax30.offset6/files_info.json', batch_size=batch_size, shuffle=True, size_scaler=1)
    # dg_val = DataGenerator('data/chiron/lambda/eval/all/samples.rawmax200.evmax30.offset6/files_info_val.json', batch_size=batch_size, shuffle=False)


    basecaller = Basecaller(
        units=units, batch_sz=batch_size, output_text_processor=dm.output_text_processor,
        input_data_type=data_type, input_padding_value=dm.input_padding_value,
        embedding_dim=embedding_dim, rnn_type=rnn_type, attention_type=attention_type,
        teacher_forcing=teacher_forcing
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5),
    )

    batch_loss = utils.BatchLogs('loss')

    # Callbacks

    checkpoint_filepath = f'models/s2s/model.{name}.{"{epoch:02d}"}/model_chp'
    # print('New checkpoints:', checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        f'info/s2s/csvlog.s2s.{name}.log', append=False
    )

    start = timer()
    hist = basecaller.fit(
        train_ds,
        epochs=epochs,
        callbacks=[batch_loss, csv_logger, model_checkpoint_callback],
        validation_data=val_ds
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

    # info['batch_loss'] = batch_loss.logs

    # with open(f'info/chiron/80/info.1.{name}.json', 'w') as info_file:
    #     json.dump(info, info_file, indent=2)



if __name__ == '__main__':
    basic_params = {
        'data_type': '',
        'batch_size': 128,
        'raw_max_len': 200,
        'event_max_len': 30,
        'units': 128,
        'epochs': 99,
        'patience': 99,
        'bases_offset': 3,
        'rnn_type': '',
        'attention_type': 'luong',
        'embedding_dim': 'one_hot',
        'event_detection': True
    }

    variants = [
        {'data_type': 'raw', 'rnn_type': 'bilstm'},
    ]

    for v in variants:
        run_chiron_training(
            **{**basic_params, **v}
        )