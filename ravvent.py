import numpy as np
from enc_dec_attn import *
from data_loader import DataModule
from basecaller import Basecaller
import json
from timeit import default_timer as timer

EMBEDDING_DIM = 1
UNITS = 32
EPOCHS = 100
PATIENCE = 50
DATA_PATH = 'data/sim_out.dep.CM000663.l200000.s0'
BASES_OFFSET = 1
DATA_TYPE = 'raw'
TEACHER_FORCING = True

NAME_SPEC = f'raw.u{UNITS}.inmax{INPUT_MAX_LEN}.b{BATCH_SIZE}.ep{EPOCHS}.pat{PATIENCE}.tf{int(TEACHER_FORCING)}'


if __name__ == '__main__':
    dm = DataModule(DATA_PATH, INPUT_MAX_LEN, BASES_OFFSET, BATCH_SIZE, DATA_TYPE, random_seed=RANDOM_SEED)

    train_ds, val_ds, test_ds = dm.get_train_val_test_split_datasets()

    print('TRAIN SIZE', tf.data.experimental.cardinality(train_ds).numpy())
    print('VALIDATION SIZE', tf.data.experimental.cardinality(val_ds).numpy())
    print('TEST SIZE', tf.data.experimental.cardinality(test_ds).numpy())

    train_basecaller = TrainBasecaller(
        UNITS,
        EMBEDDING_DIM,
        dm.output_text_processor,
        DATA_TYPE,
        dm.input_padding_value,
        teacher_forcing=TEACHER_FORCING,
        use_tf_function=True
    )

    # Configure the loss and optimizer
    train_basecaller.compile(
        optimizer=tf.optimizers.Adam(),
        loss=MaskedLoss(),
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
        encoder=train_basecaller.encoder,
        decoder=train_basecaller.decoder,
        input_padding_value=dm.input_padding_value,
        output_text_processor=dm.output_text_processor
    )

    mid_2 = timer()
    accuracies = []
    for input_batch, target_batch in test_ds:
        acc = bc.evaluate_batch((input_batch, target_batch))
        accuracies.append(acc.numpy())

    test_accuracy = np.mean(accuracies)
    end = timer()
    print(test_accuracy)

    info['test_accuracy'] = test_accuracy

    info['train_time'] = mid_1 - start
    info['test_time'] = end - mid_2

    with open(f'info/info.{NAME_SPEC}.json', 'w') as info_file:
        json.dump(info, info_file)
