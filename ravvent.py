import tensorflow as tf
from timeit import default_timer as timer
import json

from data_loader import DataModule
from basecaller import Basecaller
import utils

DATA_TYPE = 'joint'

BATCH_SIZE = 64

RAW_MAX_LEN = 100
EVENT_MAX_LEN = 30

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

UNITS = 16
EPOCHS = 2
PATIENCE = 50
DATA_PATH = 'data/chiron/train'
BASES_OFFSET = 1
TEACHER_FORCING = False
RNN_TYPE = 'gru'
LOAD_SOURCE = 'chiron'

RANDOM_SEED = 22

if DATA_TYPE == 'joint':
    NAME_MAX_LEN = f'rawmax{RAW_MAX_LEN}.evmax{EVENT_MAX_LEN}'
else:
    NAME_MAX_LEN = f'rawmax{RAW_MAX_LEN}' if DATA_TYPE == 'raw' else f'evmax{EVENT_MAX_LEN}'

NAME_SPEC = f'{DATA_TYPE}.{RNN_TYPE}.u{UNITS}.{LOAD_SOURCE}.{NAME_MAX_LEN}.b{BATCH_SIZE}.ep{EPOCHS}.pat{PATIENCE}.tf{int(TEACHER_FORCING)}'


tf.random.set_seed(RANDOM_SEED)

if __name__ == '__main__':
    # chiron 100
    dm = DataModule(
        dir='data/chiron/train/ecoli_0001_0100',
        max_raw_length=RAW_MAX_LEN,
        max_event_length=EVENT_MAX_LEN,
        bases_offset=BASES_OFFSET,
        batch_size=BATCH_SIZE,
        train_size=1,
        val_size=0,
        test_size=0,
        load_source=LOAD_SOURCE,
        random_seed=RANDOM_SEED,
        verbose=True
    )
    dm.setup()
    train_ds = dm.dataset_train

    dm.dir = 'data/chiron/eval/ecoli_eval_0001_0100'
    dm.train_size, dm.val_size, dm.test_size = 0, 0.25, 0.75
    dm.setup()
    val_ds, test_ds = dm.dataset_val, dm.dataset_test

    # # simulator
    # dm = DataModule(
    #     dir=DATA_PATH,
    #     max_raw_length=RAW_MAX_LEN,
    #     max_event_length=EVENT_MAX_LEN,
    #     bases_offset=BASES_OFFSET,
    #     batch_size=BATCH_SIZE,
    #     load_source=LOAD_SOURCE,
    #     random_seed=RANDOM_SEED,
    #     verbose=True
    # )
    # dm.setup()
    # train_ds, val_ds, test_ds = dm.dataset_train, dm.dataset_val, dm.dataset_test

    basecaller = Basecaller(
        units=UNITS,
        output_text_processor=dm.output_text_processor,
        input_data_type=DATA_TYPE,
        input_padding_value=dm.input_padding_value,
        rnn_type=RNN_TYPE,
        teacher_forcing=TEACHER_FORCING
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(),
        loss=utils.MaskedLoss(basecaller.output_padding_token),
    )
    batch_loss = utils.BatchLogs('batch_loss')

    checkpoint_filepath = f'models/model.{NAME_SPEC}/model_chp'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_batch_loss',
        mode='min',
        save_best_only=True
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_batch_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )

    start = timer()
    hist = basecaller.fit(train_ds, epochs=EPOCHS, callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback], validation_data=val_ds)
    mid_1 = timer()
    print(hist.history)

    info = {}
    info['train_history'] = hist.history

    mid_2 = timer()
    test_accuracy = basecaller.evaluate_test(test_ds)
    end = timer()
    print(test_accuracy)

    info['test_accuracy'] = test_accuracy

    info['train_time'] = mid_1 - start
    info['test_time'] = end - mid_2

    with open(f'info/info.{NAME_SPEC}.json', 'w') as info_file:
        json.dump(info, info_file)
