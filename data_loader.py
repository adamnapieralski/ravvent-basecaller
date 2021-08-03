import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle

from ont_fast5_api.fast5_interface import get_fast5_file

import utils


INPUT_MASK = tf.float32.max

class DataModule():
    def __init__(self, dir: str, max_raw_length: int, max_event_length:int, bases_offset: int = 1, batch_size: int = 64, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1, load_source: str = 'simulator', random_seed: int = 0, verbose: bool = False):
        '''Initialize DataModule
            Parameters:
                dir (str): directory to load data from (dependent on load_source)
                max_raw_length (int):
                max_event_length (int):
                bases_offset (int): Number of bases between following data rows
                batch_size (int): Number of data rows in batch
                data_type (str): Data type - allowed values: "raw", "event"
                load_source (str): supported 'simulator'/'chiron'
                random_seed (int): Random seed
        '''
        self.dir = dir
        self.max_raw_length = max_raw_length
        self.max_event_length = max_event_length
        self.bases_offset= bases_offset
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.load_source = load_source
        self.random_seed = random_seed if random_seed != 0 else np.random.randint(1)
        self.verbose = verbose

        self.scalers = {
            'raw': StandardScaler(),
            'event': {
                'mean': StandardScaler(),
                'std': StandardScaler(),
                'length': StandardScaler(),
                'd_mean': StandardScaler(),
                'sq_mean': StandardScaler(),
            }
        }

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.output_text_processor = preprocessing.TextVectorization(
            standardize=text_lower_and_start_end,
        )
        self.output_text_processor.adapt(['A C G T'])

        self.input_padding_value = INPUT_MASK

    def load_data_samples(self):
        if self.load_source == 'simulator':
            raw_aligned_data, events_sequence, bases_sequence = self._load_simulator_data(self.dir)
            raw_samples, event_samples, bases_samples = zip(*self.samples_generator(raw_aligned_data, events_sequence, bases_sequence, self.max_raw_length, self.bases_offset))
        elif self.load_source == 'chiron':
            raw_samples, event_samples, bases_samples = self._load_all_chiron_data_samples_from_dir(self.dir, 100)

        return raw_samples, event_samples, bases_samples

    def process_and_split_data_samples(self, raw_samples, event_samples, bases_samples):
        raw_data, event_data = self.pad_input_data(raw_samples, event_samples)
        raw_data = np.reshape(raw_data, (len(raw_data), self.max_raw_length, 1))
        raw_data_split, event_data_split, bases_data_split = self.train_val_test_split(raw_data, event_data, bases_samples, train_size=self.train_size, val_size=self.val_size, test_size=self.test_size)

        # scaling
        if raw_data_split[0] is not None and event_data_split[0] is not None:
            self.fit_scalers(raw_data_split[0], event_data_split[0])
        raw_data_split, self.event_data_split = self.scale_input_data(raw_data_split, event_data_split)

        bases_data_split = self.prepare_bases_data(bases_data_split)

        return raw_data_split, event_data_split, bases_data_split

    def save_to_datasets(self, raw_data_split, event_data_split, bases_data_split):
        if self.train_size > 0:
            self.dataset_train = tf.data.Dataset.from_tensor_slices((raw_data_split[0], event_data_split[0], bases_data_split[0])).batch(self.batch_size, drop_remainder=True)
        else:
            self.dataset_train = None
        if self.val_size > 0:
            self.dataset_val = tf.data.Dataset.from_tensor_slices((raw_data_split[1], event_data_split[1], bases_data_split[1])).batch(self.batch_size, drop_remainder=True)
        else:
            self.dataset_val = None
        if self.test_size > 0:
            self.dataset_test = tf.data.Dataset.from_tensor_slices((raw_data_split[2], event_data_split[2], bases_data_split[2])).batch(self.batch_size, drop_remainder=True)
        else:
            self.dataset_test = None

    def setup(self):
        raw_samples, event_samples, bases_samples = self.load_data_samples()
        raw_data_split, event_data_split, bases_data_split = self.process_and_split_data_samples(raw_samples, event_samples, bases_samples)
        self.save_to_datasets(raw_data_split, event_data_split, bases_data_split)

        if self.verbose:
            print('Max bases seq. length: {}'.format(max(map(len, bases_samples))))
            print('Train samples:\t{}, batches:\t{}'.format(0 if bases_data_split[0] == None else len(bases_data_split[0]), 0 if self.dataset_train == None else tf.data.experimental.cardinality(self.dataset_train).numpy()))
            print('Val samples:\t{}, batches:\t{}'.format(0 if bases_data_split[1] == None else len(bases_data_split[1]), 0 if self.dataset_val == None else tf.data.experimental.cardinality(self.dataset_val).numpy()))
            print('Train samples:\t{}, batches:\t{}'.format(0 if bases_data_split[2] == None else len(bases_data_split[2]), 0 if self.dataset_test == None else tf.data.experimental.cardinality(self.dataset_test).numpy()))

    def train_val_test_split(self, raw_data, event_data, bases_data, train_size=0.8, val_size=0.1, test_size=0.1):
        raw_train, raw_val, raw_test = utils.train_val_test_split(raw_data, val_size=val_size, train_size=train_size, test_size=test_size, random_state=self.random_seed, shuffle=True)
        event_train, event_val, event_test = utils.train_val_test_split(event_data, val_size=val_size, train_size=train_size, test_size=test_size, random_state=self.random_seed, shuffle=True)
        bases_train, bases_val, bases_test = utils.train_val_test_split(bases_data, val_size=val_size, train_size=train_size, test_size=test_size, random_state=self.random_seed, shuffle=True)
        return (raw_train, raw_val, raw_test), (event_train, event_val, event_test), (bases_train, bases_val, bases_test)

    def pad_input_data(self, raw_data, event_data):
        raw_padded = pad_sequences(raw_data, maxlen=self.max_raw_length, dtype='float32', padding='post', truncating='post', value=self.input_padding_value)
        event_padded = pad_sequences(event_data, maxlen=self.max_event_length, dtype='float32', padding='post', truncating='post', value=self.input_padding_value)
        return raw_padded, event_padded

    def fit_scalers(self, raw_train, event_train, partial=False):
        if partial:
            self.scalers['raw'].partial_fit((raw_train[raw_train != self.input_padding_value]).reshape(-1, 1))
            for id, feat in enumerate(['mean', 'std', 'length', 'd_mean', 'sq_mean']):
                vals = event_train[:,:,id]
                self.scalers['event'][feat].partial_fit((vals[vals != self.input_padding_value]).reshape(-1, 1))
        else:
            self.scalers['raw'].fit((raw_train[raw_train != self.input_padding_value]).reshape(-1, 1))
            for id, feat in enumerate(['mean', 'std', 'length', 'd_mean', 'sq_mean']):
                vals = event_train[:,:,id]
                self.scalers['event'][feat].fit((vals[vals != self.input_padding_value]).reshape(-1, 1))

    def scale_input_data(self, raw_data_split, event_data_split):
        raw_scaled = []
        for raw in raw_data_split:
            if raw is None:
                raw_scaled.append(None)
                continue
            raw_no_pad = raw[raw != self.input_padding_value]
            raw_no_pad_scaled = self.scalers['raw'].transform(raw_no_pad.reshape(-1, 1))
            raw[raw != self.input_padding_value] = raw_no_pad_scaled.flatten()
            raw_scaled.append(raw)

        event_scaled = []
        for event in event_data_split:
            if event is None:
                event_scaled.append(None)
                continue
            for id, feat in enumerate(['mean', 'std', 'length', 'd_mean', 'sq_mean']):
                feat_vals = event[:,:,id]
                feat_vals_no_pad = feat_vals[feat_vals != self.input_padding_value]
                feat_vals_no_pad_scaled = self.scalers['event'][feat].transform(feat_vals_no_pad.reshape(-1, 1))
                feat_vals[feat_vals != self.input_padding_value] = feat_vals_no_pad_scaled.flatten()
                event[:,:,id] = feat_vals
            event_scaled.append(event)
        return tuple(raw_scaled), tuple(event_scaled)

    def prepare_bases_data(self, bases_data_split):
        bases_prep = []
        for bases in bases_data_split:
            if bases is None:
                bases_prep.append(None)
                continue
            bases_prep.append([' '.join(bases_sample) for bases_sample in bases])
        return tuple(bases_prep)

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

        return np.array(events_sequence)

    def _load_simulator_data(self, dir):
        dir = Path(dir)
        fasta_path = dir / "sampled_read.fasta"
        fast5_path = next((dir / "fast5").iterdir())
        align_path = next((dir / "align").iterdir())

        bases_sequence = self._get_fasta_seq(fasta_path)
        raw_signal_data = self._get_fast5_raw_data(fast5_path)
        alignment_data = self._get_alignment_data(align_path)

        bases_raw_aligned_data = self._get_bases_raw_aligned_data(alignment_data, raw_signal_data)

        events_sequence = self._get_events_sequence(bases_raw_aligned_data)

        return bases_raw_aligned_data, events_sequence, bases_sequence

    def save_scalers(self, path: str):
        """Save scalers as pickle to file in path
        """
        with open(path, 'wb') as scalers_file:
            pickle.dump(self.scalers, scalers_file)

    def load_scalers(self, path: str):
        """Load scalers from file pickle in path
        """
        with open(path, 'rb') as scalers_file:
            self.scalers = pickle.load(scalers_file)

    ### chiron load source processing

    def _load_all_chiron_data_samples_from_dir(self, dir, max_files=None):
        dir = Path(dir)
        signals_paths = [p for p in dir.iterdir() if p.suffix == '.signal']
        signals_paths.sort()
        labels_paths = [p for p in dir.iterdir() if p.suffix == '.label']
        labels_paths.sort()
        if max_files is not None:
            signals_paths = signals_paths[0:max_files]
            labels_paths = labels_paths[0:max_files]

        raw_samples_all, event_samples_all, bases_samples_all = [], [], []

        for signal_path, label_path in zip(signals_paths, labels_paths):
            bases_raw_aligned_data, events_sequence, bases_sequence = self._load_chiron_single_data(signal_path, label_path)
            raw_samples, event_samples, bases_samples = zip(*self.samples_generator(bases_raw_aligned_data, events_sequence, bases_sequence, self.max_raw_length, self.bases_offset))
            raw_samples_all.extend(raw_samples)
            event_samples_all.extend(event_samples)
            bases_samples_all.extend(bases_samples)

        return raw_samples_all, event_samples_all, bases_samples_all


    def _load_chiron_single_data(self, signal_path, label_path):
        signal = np.loadtxt(signal_path)
        labels = np.loadtxt(label_path, dtype='object')
        ranges_ids = labels[:,0:2].astype('int')
        bases_sequence = labels[:,2]

        bases_raw_aligned_data = []

        for base_range in ranges_ids:
            bases_raw_aligned_data.append(signal[base_range[0]:base_range[1]])

        events_sequence = self._get_events_sequence(bases_raw_aligned_data)

        return bases_raw_aligned_data, events_sequence, bases_sequence


    def save_chiron_padded_samples(self, dir, scalers_partial_fit=False):
        dir = Path(dir)
        samples_dir = dir / f'samples.rawmax{self.max_raw_length}.evmax{self.max_event_length}.offset{self.bases_offset}'
        samples_dir.mkdir(parents=True, exist_ok=True)

        signals_paths = [p for p in dir.iterdir() if p.suffix == '.signal']
        signals_paths.sort()
        labels_paths = [p for p in dir.iterdir() if p.suffix == '.label']
        labels_paths.sort()

        random_state = self.random_seed
        max_lens = []

        for signal_path, label_path in zip(signals_paths, labels_paths):
            bases_raw_aligned_data, events_sequence, bases_sequence = self._load_chiron_single_data(signal_path, label_path)
            raw_samples, event_samples, bases_samples = zip(*self.samples_generator(bases_raw_aligned_data, events_sequence, bases_sequence, self.max_raw_length, self.bases_offset))

            if self.verbose:
                max_len = max(map(len, bases_samples))
                max_lens.append(max_len)
                print('{}: max bases seq. length: {}'.format(signal_path.stem, max_len))
                if max_len > self.max_event_length:
                    raise Exception('Max event length smaller than max length.')

            raw_samples, event_samples = self.pad_input_data(raw_samples, event_samples)
            raw_samples = np.reshape(raw_samples, (len(raw_samples), self.max_raw_length, 1))
            raw_samples, event_samples, bases_samples = sklearn_shuffle(raw_samples, event_samples, bases_samples, random_state=random_state)

            if scalers_partial_fit:
                self.fit_scalers(raw_samples, event_samples, partial=True)

            random_state += 1

            with open(samples_dir / f'{signal_path.stem}.pkl', 'wb') as f:
                pickle.dump((raw_samples, event_samples, bases_samples), f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            print('Max len: {}'.format(max(max_lens)))

    def transform_and_replace_chiron_saved_samples(self, samples_dir):
        samples_dir = Path(samples_dir)
        pkl_paths = [p for p in dir.iterdir() if p.suffix == '.pkl']
        pkl_paths.sort()

        for pkl_path in pkl_paths:
            with open(pkl_path, 'w+b') as f:
                if self.verbose:
                    print('Replacing {}'.format(pkl_path))
                (raw_samples, event_samples, bases_samples) = pickle.load(f)
                raw_data, event_data =  self.scale_input_data((raw_samples), (event_samples))
                bases_data = self.prepare_bases_data((bases_samples))
                pickle.dump((raw_data[0], event_data[0], bases_data[0]), f, protocol=pickle.HIGHEST_PROTOCOL)


def text_lower_and_start_end(text):
    text = tf.strings.lower(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text