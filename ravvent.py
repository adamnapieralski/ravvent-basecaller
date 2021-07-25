from enc_dec_attn import *
from data_loader import DataModule
from basecaller import Basecaller
import pickle

EMBEDDING_DIM = 1
UNITS = 16
EPOCHS = 4
DATA_PATH = 'data/seq_2_5k/perfect'


if __name__ == '__main__':
    dm = DataModule(DATA_PATH, INPUT_MAX_LEN, 4, BATCH_SIZE)

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

    hist = train_basecaller.fit(train_ds, epochs=EPOCHS, callbacks=[batch_loss, model_checkpoint_callback, early_stopping_callback], validation_data=val_ds)
    print(hist.history)

    with open('train_history.pickle', 'wb') as hf:
        pickle.dump(hist.history, hf, pickle.HIGHEST_PROTOCOL)

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

    with open('accuracy.pickle', 'wb') as af:
        pickle.dump(test_accuracy, af, pickle.HIGHEST_PROTOCOL)
