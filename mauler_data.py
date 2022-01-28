import numpy as np
import tensorflow as tf
import pickle
import json

from pathlib import Path
from sklearn.preprocessing import StandardScaler



from keras.preprocessing.text import Tokenizer

tk = Tokenizer(filters='', lower=True, char_level=True)
tk.fit_on_texts(['$acgt^'])
end_token = tk.texts_to_sequences('$')[0][0] - 1
start_token = tk.texts_to_sequences('^')[0][0] - 1

RANDOM_SEED = 22

def save_chiron_samples_dir(dir, verbose=True):
    dir = Path(dir)
    samples_dir = dir / f'samples.mauler'
    samples_dir.mkdir(parents=True, exist_ok=True)

    signals_paths = [p for p in dir.iterdir() if p.suffix == '.signal']
    signals_paths.sort()
    labels_paths = [p for p in dir.iterdir() if p.suffix == '.label']
    labels_paths.sort()

    files_info = [] # for use in DataGenerator

    # with open(samples_dir / 'files_info.json', 'rt') as fi:
    #     files_info = json.load(fi)

    for signal_path, label_path in zip(signals_paths, labels_paths):
        # if signal_path.stem in [e['path'].split('/')[3][0:-4] for e in files_info]:
        #     continue

        sig_samples, nuc_samples = get_chiron_single_data(signal_path, label_path)

        if verbose:
            print('{}'.format(signal_path.stem))

        # with open(dat_file_path, 'wb') as f:
        #     pickle.dump((sig_samples, nuc_samples), f, protocol=pickle.HIGHEST_PROTOCOL)

        files_info.append({
            'signal_path': signal_path.as_posix(),
            'label_path': label_path.as_posix(),
            'samples': sig_samples.shape[0]
        })

        with open(samples_dir / 'files_info.json', 'wt') as fi:
            json.dump(files_info, fi, indent=2)

    with open(samples_dir / 'files_info.json', 'wt') as fi:
        json.dump(files_info, fi, indent=2)

def split_strided(a, L, S):
    # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]

def get_chiron_single_data(signal_path, label_path):
    signal = np.loadtxt(signal_path)
    labels = np.loadtxt(label_path, dtype='object')
    ranges_ids = labels[:,0:2].astype('int')
    nuc_all = labels[:,2]

    sc = StandardScaler()
    # replace_outliers(signal)
    signal = sc.fit_transform(signal.reshape(-1, 1)).flatten()

    signal = signal[ranges_ids[0][0]:ranges_ids[-1][1]]
    signal = np.reshape(signal, (len(signal), 1))

    signal_nuc_align_gt = []
    for i in range(len(ranges_ids)):
        signal_nuc_align_gt.extend([i] * (ranges_ids[i][1] - ranges_ids[i][0]))
    signal_nuc_align_gt = np.array(signal_nuc_align_gt)

    signal_samples = split_strided(signal, 300, 30)

    signal_bases_align_gt = split_strided(signal_nuc_align_gt, 300, 30)

    max_nucs_len = 40
    nuc_samples = []
    for i in range(len(signal_bases_align_gt)):
        s = nuc_all[np.unique(signal_bases_align_gt[i])][:max_nucs_len]
        nuc_samples.append(('^' + ''.join(s)) + '$' * (max_nucs_len + 1 - len(s)))

    nuc_samples = np.array(tk.texts_to_sequences(nuc_samples))
    nuc_samples -= 1

    return signal_samples, nuc_samples

def replace_outliers(raw_data):
    out_idx = np.where(np.logical_or(raw_data < 0, raw_data > 2000))
    raw_data[out_idx] = int(np.mean(np.delete(raw_data, out_idx)))

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, files_info_path, batch_size=64, shuffle=True, size_scaler=1):
        """
        file_info_path (str): json file including array of pickle data files and number of samples in them (format: [{'path', 'samples'},])
        batch_size (int): batch size
        shuffle (bool): should files and batches inside them be shuffled between epochs
        """
        self.batch_size = batch_size
        self.files_info_path = files_info_path
        self.shuffle = shuffle
        self.fetch_ids = None
        self.read_file = None

        self.last_file_id = None
        self.files_info = None

        self.size_scaler = size_scaler

        with open(self.files_info_path, 'r') as fi:
            self.files_info = json.load(fi)

        self.fetch_ids = self._compute_new_fetch_ids()

    def _compute_new_fetch_ids(self):
        files_ids = np.arange(len(self.files_info))

        if self.size_scaler < 1:
            files_ids = files_ids[0:int(self.size_scaler * len(files_ids))]

        if self.shuffle:
            np.random.shuffle(files_ids)

        fetch_ids = []
        for f_id in files_ids:
            samples = self.files_info[f_id]['samples']
            batches_num = samples // self.batch_size
            start_ids = np.arange(0, self.batch_size * batches_num, self.batch_size)

            if self.shuffle:
                np.random.shuffle(start_ids)

            fetch_ids.extend(
                [(f_id, sid, sid+self.batch_size) for sid in start_ids]
            )
        return np.array(fetch_ids, dtype='int')

    def __getitem__(self, index):
        """Generate one batch of data
        """

        # if fetched file id changes
        if self.fetch_ids[index][0] != self.last_file_id:
            self.read_file = get_chiron_single_data(self.files_info[self.fetch_ids[index][0]]['signal_path'], self.files_info[self.fetch_ids[index][0]]['label_path'])
            self.last_file_id = self.fetch_ids[index][0]
            # print('reading ' + self.files_info[self.fetch_ids[index][0]]['signal_path'])
            # print(index)
        return (
            tf.convert_to_tensor(self.read_file[0][self.fetch_ids[index][1]:self.fetch_ids[index][2]]), # raw
            tf.convert_to_tensor(self.read_file[1][self.fetch_ids[index][1]:self.fetch_ids[index][2]]), # nuc
        )

    def __len__(self):
        return len(self.fetch_ids)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle:
            self.fetch_ids = self._compute_new_fetch_ids()
