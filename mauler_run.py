import tensorflow as tf
from timeit import default_timer as timer
import numpy as np

import mauler_data as md
import mauler as mauler
import utils

RANDOM_SEED = 22

def run():
    tf.random.set_seed(RANDOM_SEED)

    teacher_forcing = True
    learning_rate = 0.0001
    encoder_depth = 2
    encoder_units = 100
    decoder_units = 200

    steps_per_epoch = 10000

    batch_size = 128

    # name_spec = f'{data_type}.train_as_val_0.25.lr{round(learning_rate, 6)}.{rnn_type}.encu{encoder_units}.encd{encoder_depth}.decu{decoder_units}.decd{decoder_depth}.{load_source}.{name_max_len}.b{batch_size}.ep{epochs}.pat{patience}.emb{embedding_dim}.ed{int(event_detection)}.{attention_type}.tf{int(teacher_forcing) if type(teacher_forcing) is bool else round(teacher_forcing, 2)}.boff{bases_offset}'

    name = 'mauler'
    print('RUNNING', name)

    # dm.setup()
    # train_ds = dm.dataset_train

    # dm.dir = 'data/simulator/reduced/seq.4096.600000.4096.eval'
    # dm.train_size, dm.val_size, dm.test_size = 0, 0.25, 0.75
    # dm.setup()
    # val_ds, test_ds = dm.dataset_val, dm.dataset_test

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

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    #     # Everything that creates variables should be under the strategy scope.
    #     # In general this is only model construction & `compile()`.

    basecaller = mauler.Basecaller(
        enc_units=encoder_units, dec_units=decoder_units, batch_sz=batch_size,
        encoder_depth=encoder_depth, attention_type='luong',
        teacher_forcing=teacher_forcing
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5),
    )

    batch_loss = utils.BatchLogs('loss')

    # Callbacks

    checkpoint_filepath = f'models/mauler/model.run_2.replaced_outliers.{"{epoch:02d}"}/model_chp'
    print('New checkpoints:', checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        f'info/mauler/csvlog.run_2.replaced_outliers.log', append=False
    )

    dg_train = md.DataGenerator('data/chiron/lambda/train/all/samples.mauler/files_info.json', batch_size=128, shuffle=True, size_scaler=1)
    dg_val = md.DataGenerator('data/chiron/lambda/eval/all/samples.mauler/files_info.json', batch_size=128, shuffle=True, size_scaler=1)

    # def train_generator():
    #     multi_enqueuer = tf.keras.utils.OrderedEnqueuer(dg_train, use_multiprocessing=True)
    #     multi_enqueuer.start(workers=10, max_queue_size=10)
    #     while True:
    #         (batch_xs, batch_ys), dset_index = next(multi_enqueuer.get()) # I have three outputs
    #         yield (batch_xs, batch_ys), dset_index

    # dataset_train = tf.data.Dataset.from_generator(train_generator,
    #                                         output_types=(tf.float64, tf.int64, tf.int64),
    #                                         output_shapes=(tf.TensorShape([None, None, None, None]),
    #                                                         tf.TensorShape([None, None, None, None]),
    #                                                         tf.TensorShape([None, None])))

    # train_dist_dataset = strategy.experimental_distribute_dataset(dataset_train)

    hist = basecaller.fit(
        dg_train,
        epochs=30,
        callbacks=[batch_loss, model_checkpoint_callback, csv_logger],
        validation_data=dg_val,
        shuffle=False,
        steps_per_epoch=steps_per_epoch,
        validation_steps=steps_per_epoch,
    )

    info = {}
    info['train_history'] = hist.history



if __name__ == '__main__':
    run()
