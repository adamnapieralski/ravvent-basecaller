import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pathlib import Path
from sklearn.preprocessing import minmax_scale

from ont_fast5_api.fast5_interface import get_fast5_file

INPUT_MASK = -1

class DataModule():
  def __init__(self, dir: str, max_raw_length: int, bases_offset: int, batch_size: int, random_seed: int = 0):
    self.dir = dir
    self.max_raw_length = max_raw_length
    self.bases_offset= bases_offset
    self.batch_size = batch_size
    self.dataset = None
    self.random_seed = random_seed if random_seed != 0 else np.random.randint(1)
    self.output_text_processor = preprocessing.TextVectorization(
      standardize=text_lower_and_start_end,
    )

    self.setup()

  def setup(self):
    self.output_text_processor.adapt(['A C G T'])

    raw_aligned_data, bases_sequence = self._load_simulator_data(self.dir)
    raw_data, bases_data = zip(*self.samples_generator(raw_aligned_data, bases_sequence, self.max_raw_length, self.bases_offset))

    raw_prep = self.prepare_raw_data_for_dataset(raw_data)
    bases_prep = self.prepare_bases_data_for_dataset(bases_data)
    print("MAX LEN", len(max(bases_prep)))

    self.dataset = tf.data.Dataset.from_tensor_slices((raw_prep, bases_prep)).shuffle(len(raw_prep), seed=self.random_seed)
    self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)

  def prepare_raw_data_for_dataset(self, raw_data):
    raw_prep = pad_sequences(raw_data, self.max_raw_length, dtype='float32', padding='post', value=INPUT_MASK)
    # raw_prep = np.reshape(raw_prep, (len(raw_prep), self.max_raw_length, 1))
    # raw_prep = [[val] for seq in raw_prep for val in seq]
    # print(np.array(raw_prep).shape)
    return raw_prep

  def prepare_bases_data_for_dataset(self, bases_data):
    bases_prep = [' '.join(bases_sample) for bases_sample in bases_data]
    return bases_prep

  def samples_generator(self, raw_aligned_data, bases_sequence, max_raw_length, bases_offset):
    for i in range(0, len(bases_sequence) - bases_offset + 1, bases_offset):
      j = i
      raw_length_sum = len(raw_aligned_data[j])

      while raw_length_sum < max_raw_length and j < len(raw_aligned_data) - 1:
        j += 1
        raw_length_sum += len(raw_aligned_data[j])

      if raw_length_sum > max_raw_length:
        j -= 1
      elif j == len(raw_aligned_data) - 1 and raw_length_sum != max_raw_length:
        break

      # flatten raw vals
      raw_subsequence = [val for raw_single_base in raw_aligned_data[i:j+1] for val in raw_single_base]
      bases_subsequence = bases_sequence[i:j+1]

      yield raw_subsequence, bases_subsequence


  def _get_fast5_raw_data(self, fast5_path: str) -> np.ndarray:
    """Get raw data read from fast5 file (first, in case of multiple)"""
    raw_data = None

    with get_fast5_file(fast5_path, mode="r") as f:
      for read in f.get_reads():
        raw_data = read.get_raw_data()
        break

    return raw_data


  def _get_fasta_seq(self, fasta_path: str) -> np.ndarray:
    """
    Get first bases_num bases of fasta sequence
    if bases_num == -1 return whole sequence
    """
    sequence = np.array([], dtype="<U1")

    with open(fasta_path, "r") as f:
      for i, line in enumerate(f):
        if i == 1:
          sequence = np.array(list(line.rstrip()))
          break

    return sequence


  def _get_alignment_data(self, ali_path: str) -> np.ndarray:
    """
    Get simulator output alignment and return as a (2, len(raw_data)) array
    Matching raw data ([0] row) with sequence bases by id ([1] row)
    """
    data = np.loadtxt(ali_path, delimiter=" ", dtype=int)
    return data.T

  def _get_bases_raw_aligned_data(self, alignment_data, raw_data):
    """Get list of lists of base's raw values for each base/nucleotide"""
    bases_raw_aligned_data = []
    base_vals = []
    base_id = 1
    for (measurement, val) in zip(alignment_data.T, raw_data):
      if measurement[1] != base_id:
        bases_raw_aligned_data.append(base_vals)
        base_id = measurement[1]
        base_vals = []
      base_vals.append(val)
    bases_raw_aligned_data.append(base_vals)
    return bases_raw_aligned_data


  def _load_simulator_data(self, dir):
    dir = Path(dir)
    fasta_path = dir / "sampled_read.fasta"
    fast5_path = next((dir / "fast5").iterdir())
    align_path = next((dir / "align").iterdir())

    bases_sequence = self._get_fasta_seq(fasta_path)
    raw_signal_data = self._get_fast5_raw_data(fast5_path)
    alignment_data = self._get_alignment_data(align_path)

    # FIXME change this to proper scaling for subsets
    raw_signal_data = minmax_scale(raw_signal_data)

    return self._get_bases_raw_aligned_data(alignment_data, raw_signal_data), bases_sequence

def text_lower_and_start_end(text):
  text = tf.strings.lower(text)
  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text