import tensorflow as tf
from enc_dec_attn import *
from data_loader import DataModule
from basecaller import Basecaller
import json
from timeit import default_timer as timer

DATA_TYPE = 'joint'

BATCH_SIZE = 64

RAW_MAX_LEN = 150
EVENT_MAX_LEN = 40

VAL_SIZE = 0.1
TEST_SIZE = 0.1

UNITS = 16
EPOCHS = 2
PATIENCE = 50
DATA_PATH = 'data/chiron/train'
BASES_OFFSET = 1
TEACHER_FORCING = False

RANDOM_SEED = 22

if DATA_TYPE == 'joint':
    NAME_MAX_LEN = f'rawmax{RAW_MAX_LEN}.evmax{EVENT_MAX_LEN}'
else:
    NAME_MAX_LEN = f'rawmax{RAW_MAX_LEN}' if DATA_TYPE == 'raw' else f'evmax{EVENT_MAX_LEN}'

NAME_SPEC = f'{DATA_TYPE}.u{UNITS}.{NAME_MAX_LEN}.b{BATCH_SIZE}.ep{EPOCHS}.pat{PATIENCE}.tf{int(TEACHER_FORCING)}'


tf.random.set_seed(RANDOM_SEED)

if __name__ == '__main__':
    dm = DataModule(DATA_PATH, RAW_MAX_LEN, EVENT_MAX_LEN, bases_offset=BASES_OFFSET, batch_size=BATCH_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, load_source='chiron', random_seed=RANDOM_SEED)

    train_ds, val_ds, test_ds = dm.dataset_train, dm.dataset_val, dm.dataset_test

    print('TRAIN SIZE', tf.data.experimental.cardinality(train_ds).numpy())
    print('VALIDATION SIZE', tf.data.experimental.cardinality(val_ds).numpy())
    print('TEST SIZE', tf.data.experimental.cardinality(test_ds).numpy())

    train_basecaller = TrainBasecaller(
        units=UNITS,
        output_text_processor=dm.output_text_processor,
        input_data_type=DATA_TYPE,
        input_padding_value=dm.input_padding_value,
        teacher_forcing=TEACHER_FORCING,
        use_tf_function=True
    )

    # Configure the loss and optimizer
    train_basecaller.compile(
        optimizer=tf.optimizers.Adam(),
        loss=MaskedLoss(train_basecaller.output_padding_token),
    )
    batch_loss = BatchLogs('batch_loss')

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
    hist = train_basecaller.fit(train_ds, epochs=EPOCHS, callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback], validation_data=val_ds)
    mid_1 = timer()
    print(hist.history)

    info = {}
    info['train_history'] = hist.history

    bc = Basecaller(
        encoder_raw=train_basecaller.encoder_raw,
        encoder_event=train_basecaller.encoder_event,
        decoder=train_basecaller.decoder,
        input_data_type=DATA_TYPE,
        input_padding_value=dm.input_padding_value,
        output_text_processor=dm.output_text_processor
    )

    mid_2 = timer()
    test_accuracy = bc.evaluate(test_ds)
    end = timer()
    print(test_accuracy)

    info['test_accuracy'] = test_accuracy

    info['train_time'] = mid_1 - start
    info['test_time'] = end - mid_2

    with open(f'info/info.{NAME_SPEC}.json', 'w') as info_file:
        json.dump(info, info_file)
