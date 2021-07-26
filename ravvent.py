import numpy as np
from enc_dec_attn import *
from data_loader import DataModule
from basecaller import Basecaller
import json

EMBEDDING_DIM = 1
UNITS = 32
EPOCHS = 100
PATIENCE = 50
DATA_PATH = 'data/sim_out.dep.CM000663.l200000.s0'
BASES_OFFSET = 1

NAME_SPEC = f'raw.u{UNITS}.inmax{INPUT_MAX_LEN}.b{BATCH_SIZE}.ep{EPOCHS}.pat{PATIENCE}'


if __name__ == '__main__':
    dm = DataModule(DATA_PATH, INPUT_MAX_LEN, BASES_OFFSET, BATCH_SIZE)

    train_ds, val_ds, test_ds = dm.get_train_val_test_split_datasets()

    print('TRAIN SIZE', tf.data.experimental.cardinality(train_ds).numpy())
    print('VALIDATION SIZE', tf.data.experimental.cardinality(val_ds).numpy())
    print('TEST SIZE', tf.data.experimental.cardinality(test_ds).numpy())

    train_basecaller = TrainBasecaller(
        UNITS, EMBEDDING_DIM,
        output_text_processor=dm.output_text_processor,
        use_tf_function=True)

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

    hist = train_basecaller.fit(train_ds, epochs=EPOCHS, callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback], validation_data=val_ds)
    print(hist.history)

    info = {}
    info['train_history'] = hist.history

    bc = Basecaller(
        encoder=train_basecaller.encoder,
        decoder=train_basecaller.decoder,
        output_text_processor=dm.output_text_processor
    )

    accuracies = []
    for input_batch, target_batch in test_ds:
        acc = bc.evaluate_batch((input_batch, target_batch))
        accuracies.append(acc.numpy())

    test_accuracy = np.mean(accuracies)
    print(test_accuracy)

    info['test_accuracy'] = test_accuracy

    with open(f'info/info.{NAME_SPEC}.json', 'w') as info_file:
        json.dump(info, info_file)

