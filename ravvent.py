from enc_dec_attn import *
from data_loader import DataModule
from basecaller import Basecaller


if __name__ == '__main__':
  # inp = [[1.,2.,3.,4.], [2.,3.,1.,4.], [4.,4.,2.,1.], [3.,1.,4.,1.], [3.,1.,4.,1.]]
  # targ = ['a c g t', 'g t c a', 'a g a g', 'g t a c', 't t t a']

  # dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(len(inp))
  # dataset = dataset.batch(BATCH_SIZE)

  dm = DataModule('data/seq_2_5k/perfect', INPUT_MAX_LEN, 1, BATCH_SIZE)
  dataset = dm.dataset
  print(dataset)

  dm_val = DataModule('data/seq_1/perfect', INPUT_MAX_LEN, 1, BATCH_SIZE)
  val_dataset = dm_val.dataset

  embedding_dim = 1
  units = 16

  for example_input_batch, example_target_batch in dataset.take(1):
    print(example_input_batch)
    print()
    print(example_target_batch)
    break

  translator = TrainTranslator(
      units, embedding_dim,
      output_text_processor=dm.output_text_processor,
      use_tf_function=True)

  # Configure the loss and optimizer
  translator.compile(
      optimizer=tf.optimizers.Adam(),
      loss=MaskedLoss(),
  )
  batch_loss = BatchLogs('batch_loss')

  # checkpoint_filepath = 'models/chp'
  # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  #   filepath=checkpoint_filepath,
  #   save_weights_only=True,
  #   monitor='batch_loss',
  #   mode='min',
  #   save_best_only=True)

  hist = translator.fit(dataset, epochs=3, callbacks=[batch_loss], validation_data=val_dataset)
  print(hist.history)

  # raw_data, nucleotides = dl._load_simulator_data('data/seq_1/perfect')

  # bc = Basecaller(
  #   encoder=translator.encoder,
  #   decoder=translator.decoder,
  #   output_text_processor=dm.output_text_processor,
  # )

  # out = bc.tf_translate(
  #   raw_input=example_input_batch,
  # )

  # example_output_tokens = tf.random.uniform(
  #   shape=[5, 2], minval=0, dtype=tf.int64,
  #   maxval=dm.output_text_processor.vocabulary_size())
  # out = bc.tokens_to_bases_sequence(example_output_tokens).numpy()
  # print(out)
