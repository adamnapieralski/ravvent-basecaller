"""Script to find the best window_lengths_[12] parameters for event_detection
based on the results on training data
"""
import numpy as np
from pathlib import Path
import pickle

from event_detector import EventDetector
from timeit import default_timer as timer


def get_raw_and_ref_ranges_data(signal_path, label_path):
    signal = np.loadtxt(signal_path)
    labels = np.loadtxt(label_path, dtype='object')
    ranges_ids = labels[:,0:2].astype('int')

    signal = signal[ranges_ids[0,0]:ranges_ids[-1,1]]

    return signal, ranges_ids

def evaluate_event_detection(raw_data, win_len_1, win_len_2):
    ed = EventDetector(
        window_length1=win_len_1, window_length2=win_len_2)

    events = ed.run(raw_data)
    return len(events)


def evaluate_sequence(signal_path: Path, label_path: Path):
    raw, ref_ranges = get_raw_and_ref_ranges_data(signal_path, label_path)
    ref_len = ref_ranges.shape[0]

    res = {}
    st = timer()
    for wl1 in range(3, 10, 1):
        for wl2 in range(wl1 + 1 if wl1 % 2 == 0 else wl1 + 2, 22, 2):
            ed_len = evaluate_event_detection(raw, wl1, wl2)
            res[(wl1, wl2)] = ed_len

    print(ref_len, '\t', round(timer() - st, 2))
    return {
        'name': signal_path.stem,
        'ref_len': ref_len,
        'ed': res
    }

def evaluate_all_dir(dir_path, results_path):
    dir_path = Path(dir_path)
    signals_paths = [p for p in dir_path.iterdir() if p.suffix == '.signal']
    signals_paths.sort()
    labels_paths = [p for p in dir_path.iterdir() if p.suffix == '.label']
    labels_paths.sort()

    results = []

    for signal_path, label_path in zip(signals_paths, labels_paths):
        print(signal_path.stem)
        results.append(evaluate_sequence(signal_path, label_path))
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

def get_best_params(results_paths):
    data_all = []
    for res_path in results_paths:
        with open(res_path, 'rb') as f:
            data_all.append(pickle.load(f))

    results = {}
    for k in data_all[0][0]['ed'].keys():
        results[k] = []

    for data in data_all:
        for d in data:
            ref_len = d['ref_len']
            for k,v in d['ed'].items():
                results[k].append((v - ref_len) / ref_len)
    for k in results.keys():
        results[k] = np.abs(np.mean(results[k]))

    return list(results.keys())[np.argmin(list(results.values()))]

if __name__ == '__main__':
    evaluate_all_dir('../data/chiron/ecoli/train', '../data/chiron/ecoli/train/ed_param_search.pkl')
    best = get_best_params(['../data/chiron/ecoli/train/ed_param_search.pkl', '../data/chiron/lambda/train/all/ed_param_search.pkl'])
    print(best)
