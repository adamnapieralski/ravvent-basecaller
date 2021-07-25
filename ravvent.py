from enc_dec_attn import *
from data_loader import DataModule
from basecaller import Basecaller

EMBEDDING_DIM = 1
UNITS = 16


if __name__ == '__main__':
    dm = DataModule('data/seq_2_5k/perfect', INPUT_MAX_LEN, 1, BATCH_SIZE)

    train_ds, val_ds, test_ds = dm.get_train_val_test_split_datasets()

    for example_input_batch, example_target_batch in train_ds.take(1):
        print(example_input_batch)
        print()
        print(example_target_batch)
        break

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

    checkpoint_filepath = 'models/chp'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_batch_loss',
        mode='min',
        save_best_only=True
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_batch_loss',
        patience=1,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )

    hist = train_basecaller.fit(train_ds, epochs=5, callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback], validation_data=val_ds)
    print(hist.history)

    bc = Basecaller(
        encoder=train_basecaller.encoder,
        decoder=train_basecaller.decoder,
        output_text_processor=dm.output_text_processor
    )

    acc = bc.evaluate_batch((example_input_batch, example_target_batch))
    print(acc)
