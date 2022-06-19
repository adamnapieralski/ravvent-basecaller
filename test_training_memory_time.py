import tensorflow as tf
from timeit import default_timer as timer

import data_loader as dl
from basecaller import Basecaller

from memory_profiler import profile

RANDOM_SEED = 22

def run(data_type):
    tf.random.set_seed(RANDOM_SEED)

    teacher_forcing = 0.5
    learning_rate = 0.0001
    encoder_depth = 2
    encoder_units = 128
    decoder_depth = 1
    decoder_units = 128

    batch_size = 128
    epochs = 1
    stride = 6
    rnn_type = 'bilstm'
    attention_type = 'luong'

    steps_per_epoch = 30
    validation_steps = 0

    dg_train = dl.RawEventNucDataGenerator('data/chiron/lambda/train/all/files_info.snippets.stride_6.json', stride, shuffle=True, initial_random_seed=0)

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

    load_checkpoint = f'models/snippets/mask/encd_2_decd_1/model.1.{data_type}.lambda.mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.20/model_chp'
    print(f'Loading weights: {load_checkpoint}')
    basecaller.load_weights(load_checkpoint)

    @profile
    def train(data_type):
        print(data_type)
        hist = basecaller.fit(
            dg_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
            shuffle=False
        )

    start = timer()
    train(data_type)
    elapsed = timer() - start
    print(f"Training {data_type} took {elapsed}")

if __name__ == '__main__':
    run('raw')
    run('joint')
    run('event')
