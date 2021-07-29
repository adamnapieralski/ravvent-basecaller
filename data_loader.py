import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pathlib import Path
from sklearn.preprocessing import minmax_scale, scale

from ont_fast5_api.fast5_interface import get_fast5_file

INPUT_MASK = -1

class DataModule():
    def __init__(self, dir: str, max_raw_length: int, max_event_length:int, bases_offset: int, batch_size: int, data_type: str, random_seed: int = 0):
        '''Initialize DataModule
            Parameters:
                dir (str): output directory of simulator
                max_raw_length (int):
                max_event_length (int):
                bases_offset (int): Number of bases between following data rows
                batch_size (int): Number of data rows in batch
                data_type (str): Data type - allowed values: "raw", "event"
                random_seed (int): Random seed
        '''
        self.dir = dir
        self.max_raw_length = max_raw_length
        self.max_event_length = max_event_length
        self.bases_offset= bases_offset
        self.batch_size = batch_size
        self.data_type = data_type
        self.random_seed = random_seed if random_seed != 0 else np.random.randint(1)

        self.dataset = None
        self.output_text_processor = preprocessing.TextVectorization(
            standardize=text_lower_and_start_end,
        )

        self.input_padding_value = INPUT_MASK

        self.setup()

    def setup(self):
        self.output_text_processor.adapt(['A C G T'])

        raw_aligned_data, events_sequence, bases_sequence = self._load_simulator_data(self.dir)
        raw_data, events_data, bases_data = zip(*self.samples_generator(raw_aligned_data, events_sequence, bases_sequence, self.max_raw_length, self.bases_offset))

        # preparation
        raw_prep = self.prepare_raw_data_for_dataset(raw_data)
        events_prep = self.prepare_events_data_for_dataset(events_data)
        bases_prep = self.prepare_bases_data_for_dataset(bases_data)

        # verbose
        print('MAX LEN', len(max(bases_prep)), 'RAW_MAX_LEN', self.max_raw_length, 'EVENT_MAX_LEN', self.max_event_length)

        self.dataset = tf.data.Dataset.from_tensor_slices((raw_prep, events_prep, bases_prep)).shuffle(len(raw_prep), seed=self.random_seed)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)

    def get_train_val_test_split_datasets(self, val_split=0.1, test_split=0.1):
        dataset_sz = tf.data.experimental.cardinality(self.dataset).numpy()
        train_split = 1 - val_split - test_split

        train_sz = int(train_split * dataset_sz)
        val_sz = int(val_split * dataset_sz)

        train_ds = self.dataset.take(train_sz)
        remaining_ds = self.dataset.skip(train_sz)

        val_ds = remaining_ds.take(val_sz)
        test_ds = remaining_ds.skip(val_sz)

        return train_ds, val_ds, test_ds

    def prepare_raw_data_for_dataset(self, raw_data):
        raw_prep = pad_sequences(raw_data, self.max_raw_length, dtype='float32', padding='post', value=self.input_padding_value)
        raw_prep = np.reshape(raw_prep, (len(raw_prep), self.max_raw_length, 1))
        return raw_prep

    def prepare_events_data_for_dataset(self, events_data):
        events_prep = pad_sequences(events_data, self.max_event_length, dtype='float32', padding='post', truncating='post', value=self.input_padding_value)
        return events_prep

    def prepare_bases_data_for_dataset(self, bases_data):
        bases_prep = [' '.join(bases_sample) for bases_sample in bases_data]
        return bases_prep

    def samples_generator(self, raw_aligned_data, events_sequence, bases_sequence, max_raw_length, bases_offset):
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
            events_subsequence = events_sequence[i:j+1]
            bases_subsequence = bases_sequence[i:j+1]

            yield raw_subsequence, events_subsequence, bases_subsequence


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

    def _get_events_sequence(self, bases_raw_aligned_data):
        events_sequence = []
        prev_mean = 0

        for base_raw_data in bases_raw_aligned_data:
            mean = np.mean(base_raw_data)
            std = np.std(base_raw_data)
            length = len(base_raw_data)
            d_mean = mean - prev_mean
            prev_mean = mean
            sq_mean = mean ** 2
            events_sequence.append((mean, std, length, d_mean, sq_mean))

        events_sequence = np.array(events_sequence)
        # STANDARIZATION lengths
        # FIXME change this to proper scaling for subsets
        events_sequence[:, 2] = scale(events_sequence[:, 2])

        return events_sequence

    def _load_simulator_data(self, dir):
        dir = Path(dir)
        fasta_path = dir / "sampled_read.fasta"
        fast5_path = next((dir / "fast5").iterdir())
        align_path = next((dir / "align").iterdir())

        bases_sequence = self._get_fasta_seq(fasta_path)
        raw_signal_data = self._get_fast5_raw_data(fast5_path)
        alignment_data = self._get_alignment_data(align_path)

        # FIXME change this to proper scaling for subsets
        # NORMALIZATION
        raw_signal_data = minmax_scale(raw_signal_data)
        bases_raw_aligned_data = self._get_bases_raw_aligned_data(alignment_data, raw_signal_data)

        events_sequence = self._get_events_sequence(bases_raw_aligned_data)

        return bases_raw_aligned_data, events_sequence, bases_sequence

def text_lower_and_start_end(text):
    text = tf.strings.lower(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text