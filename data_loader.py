import tensorflow as tf
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pathlib import Path

from event_detection.event_detector import EventDetector


ED_WINDOW_LENGTH_1 = 6
ED_WINDOW_LENGTH_2 = 9
INPUT_PADDING = 0.

MAX_RAW_LEN = 200
MAX_EVENT_LEN = 30


nuc_tk = Tokenizer(filters='', lower=True, char_level=True, split='')
nuc_tk.word_index = {'': 0, '^': 1, '$': 2, 'a': 3, 'c': 4, 'g': 5, 't': 6}
nuc_tk.index_word = {v: k for k, v in nuc_tk.word_index.items()}

NUC_TOKEN_END = nuc_tk.word_index['^']
NUC_TOKEN_START = nuc_tk.word_index['$']
NUC_TOKEN_PAD = nuc_tk.word_index['']


def compute_fitting_event_ranges(events_lens, stride, raw_max_len=200):
    """Returns array with two elements row ([start_id, end_id (exclusive)]) of events,
    so that total lengts of events in ranges <= raw_max_len
    Length of return array is a bit smaller than initial number of events, since last ones don't
    make up enough range
    """
    cum_lens = np.cumsum(events_lens, axis=0, dtype=np.int32)

    range_ids = []
    for i in range(0, len(events_lens), stride):
        end_id = np.argmax(cum_lens > raw_max_len)
        if end_id == 0:
            break
        range_ids.append((i, end_id))
        if (i + stride - 1) >= len(cum_lens):
            break
        cum_lens -= cum_lens[i + stride - 1]
    return np.array(range_ids)

def convert_events_ranges_to_raw_ranges(events_ranges, events):
    return np.column_stack(
        (events[:,0][events_ranges[:,0]].astype(np.int32), events[:,0][events_ranges[:,1] - 1].astype(np.int32)) # -1 here relates to events_ranges being end-exclusive
    )

def convert_ranges_to_id_sequence(ranges):
    ids_lens = ranges[:,1] - ranges[:,0]
    core_id_seq = np.repeat(
        np.arange(ranges.shape[0]), ids_lens)
    if ranges[0,0] == 0:
        return core_id_seq

    return np.concatenate(
        (np.full(ranges[0,0], -1), core_id_seq)
    )

def get_subseqs_included_in_ranges(seq, ranges):
    subseqs = []
    for range in ranges:
        subseqs.append(seq[range[0]:range[1]])
    return subseqs

def prepare_snippets(raw, nuc_raw_ranges, nuc_reference_symbols, stride):
    event_detector = EventDetector(window_length1=ED_WINDOW_LENGTH_1, window_length2=ED_WINDOW_LENGTH_2)
    events = event_detector.run(raw)

    events = np.array([
        (e.start, e.end, e.length, e.mean, e.stdv, e.mean ** 2,
         e.mean - events[i - 1].mean if i != 0 else 0) for i, e in enumerate(events)])

    events_scaler = StandardScaler()
    events_scaler.fit(events[:,2:])

    # align events to reference nuc sequence
    events = events[np.logical_and(events[:,0] >= nuc_raw_ranges[0,0], events[:,1] <= nuc_raw_ranges[-1,1]), :]

    events[0,2] += events[0,0] - nuc_raw_ranges[0,0]
    events[0,0] = nuc_raw_ranges[0,0]

    events[-1,2] = nuc_raw_ranges[-1,1] - events[-1,0]

    std_sc = StandardScaler()
    raw_sc = std_sc.fit_transform(raw.reshape(-1,1))

    events_ranges = compute_fitting_event_ranges(events[:, 2], stride, raw_max_len=MAX_RAW_LEN)

    raw_ranges = convert_events_ranges_to_raw_ranges(events_ranges, events)

    events_sc = events_scaler.transform(events[:,2:])

    raw_snippets = get_subseqs_included_in_ranges(raw_sc, raw_ranges)
    event_snippets = get_subseqs_included_in_ranges(events_sc, events_ranges)

    nuc_reference_id_seq = convert_ranges_to_id_sequence(nuc_raw_ranges)
    nuc_id_snippets = get_subseqs_included_in_ranges(nuc_reference_id_seq, raw_ranges)
    nuc_id_snippets = [np.unique(id_snipp) for id_snipp in nuc_id_snippets]
    nuc_sym_snippets = [
        '$' + ''.join(nuc_reference_symbols[ids]) + '^'
        for ids in nuc_id_snippets]

    return raw_snippets, event_snippets, nuc_sym_snippets

def pad_input_snippets(snippets, maxlen):
    return pad_sequences(snippets, maxlen=maxlen, dtype='float32', padding='post', truncating='post', value=INPUT_PADDING)

def load_data_from_single_signal_label(signal_path, label_path, stride):
    raw = np.loadtxt(signal_path, dtype=int)
    label = np.loadtxt(label_path, dtype=object)
    nuc_raw_ranges = label[:, :2].astype(int)
    nuc_reference_symbols = label[:,2]

    raw_snippets, event_snippets, nuc_sym_snippets = prepare_snippets(raw, nuc_raw_ranges, nuc_reference_symbols, stride)
    raw_snippets = pad_input_snippets(raw_snippets, MAX_RAW_LEN)
    event_snippets = pad_input_snippets(event_snippets, MAX_EVENT_LEN)

    nuc_tk_snippets = nuc_tk.texts_to_sequences(nuc_sym_snippets)
    nuc_tk_snippets = pad_sequences(nuc_tk_snippets, maxlen=None, padding='post', value=NUC_TOKEN_PAD, dtype='int64')

    return raw_snippets, event_snippets, nuc_tk_snippets


def create_files_info(files_dir, stride=6, verbose=True):
    dir = Path(files_dir)
    files_info_path = dir / f'files_info.snippets.stride_{stride}.json'

    signals_paths = [p for p in dir.iterdir() if p.suffix == '.signal']
    signals_paths.sort()
    labels_paths = [p for p in dir.iterdir() if p.suffix == '.label']
    labels_paths.sort()

    files_info = [] # for use in DataGenerator

    for signal_path, label_path in zip(signals_paths, labels_paths):
        raw_snippets, _, _ = load_data_from_single_signal_label(signal_path, label_path, stride)

        if verbose:
            print('{}'.format(signal_path.stem))

        files_info.append({
            'signal_path': signal_path.as_posix(),
            'label_path': label_path.as_posix(),
            'snippets_num': raw_snippets.shape[0]
        })

        with open(files_info_path, 'wt') as fi:
            json.dump(files_info, fi, indent=2)

    with open(files_info_path, 'wt') as fi:
        json.dump(files_info, fi, indent=2)

def split_eval_files_info_into_test_validation(val_fraction: float, eval_files_info_path: str):
    """Used to create two new files_info_{val/test}.json files, which split files into desired shuffled fractions
    """
    files_info_data = []
    with open(eval_files_info_path, 'r') as f:
        files_info_data = json.load(f)

    files_random_ids = np.arange(len(files_info_data))
    np.random.shuffle(files_random_ids)

    val_files_ids = files_random_ids[:int(val_fraction * len(files_random_ids))]
    test_files_ids = files_random_ids[int(val_fraction * len(files_random_ids)):]

    val_files_info_path = eval_files_info_path.replace('eval', 'val')
    test_files_info_path = eval_files_info_path.replace('eval', 'test')

    with open(val_files_info_path, 'wt') as f:
        json.dump([files_info_data[i] for i in val_files_ids], f, indent=2)
    with open(test_files_info_path, 'wt') as f:
        json.dump([files_info_data[i] for i in test_files_ids], f, indent=2)


class RawEventNucDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, files_info_path, stride, batch_size=128, shuffle=True, initial_random_seed=0, size_scaler=1):
        """
        file_info_path (str): json file including array of info objects about data files (format: [{'signal_path', 'label_path', 'snippets_num'},])
        batch_size (int): batch size
        shuffle (bool): should files and batches inside them be shuffled between epochs
        """
        self.batch_size = batch_size
        self.stride = stride
        self.files_info_path = files_info_path
        self.shuffle = shuffle
        self.fetch_ids = None
        self.single_data_loaded = None

        self.last_file_id = None
        self.files_info = None

        self.random_seed = initial_random_seed
        self.rng = np.random.default_rng(self.random_seed)

        self.size_scaler = size_scaler

        with open(self.files_info_path, 'r') as fi:
            self.files_info = json.load(fi)

        self.fetch_ids = self._compute_new_fetch_ids()

    def _compute_new_fetch_ids(self):
        files_ids = np.arange(len(self.files_info))

        if self.size_scaler < 1:
            files_ids = files_ids[0:int(self.size_scaler * len(files_ids))]

        if self.shuffle:
            self.rng.shuffle(files_ids)

        fetch_ids = []
        for f_id in files_ids:
            snippets_num = self.files_info[f_id]['snippets_num']
            batches_num = snippets_num // self.batch_size
            start_ids = np.arange(0, self.batch_size * batches_num, self.batch_size)

            if self.shuffle:
                self.rng.shuffle(start_ids)

            fetch_ids.extend(
                [(f_id, sid, sid+self.batch_size) for sid in start_ids]
            )
        return np.array(fetch_ids, dtype='int')

    def __getitem__(self, index):
        """Generate one batch of data
        """

        # if fetched file id changes
        if self.fetch_ids[index][0] != self.last_file_id:
            self.single_data_loaded = load_data_from_single_signal_label(
                self.files_info[self.fetch_ids[index][0]]['signal_path'],
                self.files_info[self.fetch_ids[index][0]]['label_path'],
                self.stride)
            self.last_file_id = self.fetch_ids[index][0]

        return (
            tf.convert_to_tensor(self.single_data_loaded[0][self.fetch_ids[index][1]:self.fetch_ids[index][2]]), # raw
            tf.convert_to_tensor(self.single_data_loaded[1][self.fetch_ids[index][1]:self.fetch_ids[index][2]]), # event
            tf.convert_to_tensor(self.single_data_loaded[2][self.fetch_ids[index][1]:self.fetch_ids[index][2]]), # nuc
        )

    def __len__(self):
        return len(self.fetch_ids)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle:
            self.random_seed += 1
            self.rng = np.random.default_rng(self.random_seed)
            self.fetch_ids = self._compute_new_fetch_ids()


if __name__ == '__main__':
    split_eval_files_info_into_test_validation(0.25, 'data/chiron/lambda/eval/all/files_info.eval.snippets.stride_6.json')