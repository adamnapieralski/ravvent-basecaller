from enc_dec_attn import *
from data_loader import DataModule

from tensorflow.keras.layers.experimental import preprocessing

def tf_lower_and_start_end(text):
  # Split accecented characters.
  text = tf.strings.lower(text)
  # print(tf.strings.join(tf.strings.unicode_split(text, 'UTF-8').to_list()[0]))
  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

output_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_start_end
    # vocabulary=['', '[UNK]', '[START]', '[END]', 'a', 'c', 'g', 't']
)
output_text_processor.adapt(['A C G T'])



# dataset = dataset.batch(BATCH_SIZE)

if __name__ == '__main__':
  # inp = [[1.,2.,3.,4.], [2.,3.,1.,4.], [4.,4.,2.,1.], [3.,1.,4.,1.], [3.,1.,4.,1.]]
  # targ = ['a c g t', 'g t c a', 'a g a g', 'g t a c', 't t t a']

  # dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(len(inp))

  dm = DataModule('data/seq_1/perfect', INPUT_MAX_LEN, 1, BATCH_SIZE)
  dataset = dm.dataset
  print(dataset)

  embedding_dim = 4
  units = 16
  print('DATESET', dataset)

  for example_input_batch, example_target_batch in dataset.take(1):
    print(example_input_batch)
    print()
    print(example_target_batch)
    break

  translator = TrainTranslator(
      units, embedding_dim,
      output_text_processor=output_text_processor,
      use_tf_function=True)

  # Configure the loss and optimizer
  translator.compile(
      optimizer=tf.optimizers.Adam(),
      loss=MaskedLoss(),
  )
  batch_loss = BatchLogs('batch_loss')
  translator.fit(dataset, epochs=3, callbacks=[batch_loss])

  # raw_data, nucleotides = dl._load_simulator_data('data/seq_1/perfect')
