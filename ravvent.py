import tensorflow as tf
from timeit import default_timer as timer

import data_loader as dl
from basecaller import Basecaller

from memory_profiler import profile

RANDOM_SEED = 22

def run():
    tf.random.set_seed(RANDOM_SEED)

    teacher_forcing = 0.5
    learning_rate = 0.0001
    encoder_depth = 2
    encoder_units = 128
    decoder_depth = 1
    decoder_units = 128

    data_type = 'joint'
    batch_size = 128
    epochs = 40
    stride = 6
    rnn_type = 'bilstm'
    attention_type = 'luong'

    steps_per_epoch = 10000
    validation_steps = 1500

    name = f'{data_type}.lambda.mask.pad.lr{round(learning_rate, 6)}.{rnn_type}.encu{encoder_units}.encd{encoder_depth}.decu{decoder_units}.decd{decoder_depth}.b{batch_size}.{attention_type}.tf{int(teacher_forcing) if type(teacher_forcing) is bool else round(teacher_forcing, 2)}.strd{stride}.spe{steps_per_epoch}.spv{validation_steps}'

    print('RUNNING', name)

    dg_train = dl.RawEventNucDataGenerator('data/chiron/lambda/train/all/files_info.snippets.stride_6.json', stride, shuffle=True, initial_random_seed=0)
    dg_val = dl.RawEventNucDataGenerator('data/chiron/lambda/eval/all/files_info.val.snippets.stride_6.json', stride, shuffle=True, initial_random_seed=0)

    basecaller = Basecaller(
        enc_units=encoder_units,
        dec_units=decoder_units,
        batch_sz=batch_size,
        tokenizer=dl.nuc_tk,
        input_data_type=data_type,
        input_padding_value=dl.INPUT_PADDING,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        rnn_type=rnn_type,
        attention_type=attention_type,
        teacher_forcing=teacher_forcing,
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.),
    )

    # load_checkpoint = 'models/snippets/mask/encd_2_decd_2/model.3.event.lambda.mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd2.b128.luong.tf0.5.strd6.spe10000.spv1500.08/model_chp'
    # print(f'Loading weights: {load_checkpoint}')
    # basecaller.load_weights(load_checkpoint)

    checkpoint_filepath = f'models/snippets/mask/encd_{encoder_depth}_decd_{decoder_depth}/model.1.{name}.{"{epoch:02d}"}/model_chp'
    print('New checkpoints:', checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        f'info/csvlog.test.{name}.log', append=False
    )

    @profile
    def train():
        hist = basecaller.fit(
            dg_train,
            epochs=epochs,
            # callbacks=[model_checkpoint_callback, csv_logger],
            validation_data=dg_val,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            shuffle=False
        )

    train()

if __name__ == '__main__':
    run()
