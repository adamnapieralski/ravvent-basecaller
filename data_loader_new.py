import enum
import numpy as np
from sklearn.preprocessing import StandardScaler
from event_detection.event_detector import EventDetector


def compute_fitting_event_ranges(events_lens, raw_max_len=200):
    """Returns array with two elements row ([start_id, end_id (exclusive)]) of events,
    so that total lengts of events in ranges <= raw_max_len
    Length of return array is a bit smaller than initial number of events, since last ones don't
    make up enough range
    """
    cum_lens = np.cumsum(events_lens, axis=0, dtype=np.int32)

    range_ids = []
    for i in range(len(events_lens)):
        end_id = np.argmax(cum_lens > raw_max_len)
        if end_id == 0:
            break
        range_ids.append((i, end_id))
        cum_lens -= cum_lens[i]
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
    event_detector = EventDetector(window_length2=16)
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

    events_ranges = compute_fitting_event_ranges(events[:, 2], raw_max_len=200)
    events_ranges = events_ranges[0::stride]

    raw_ranges = convert_events_ranges_to_raw_ranges(events_ranges, events)

    events_sc = events_scaler.transform(events[:,2:])

    raw_snippets = get_subseqs_included_in_ranges(raw_sc, raw_ranges)
    event_snippets = get_subseqs_included_in_ranges(events_sc, events_ranges)

    nuc_reference_id_seq = convert_ranges_to_id_sequence(nuc_raw_ranges)
    nuc_id_snippets = get_subseqs_included_in_ranges(nuc_reference_id_seq, raw_ranges)
    nuc_id_snippets = [np.unique(id_snipp) for id_snipp in nuc_id_snippets]
    nuc_sym_snippets = [nuc_reference_symbols[ids] for ids in nuc_id_snippets]

    return raw_snippets, event_snippets, nuc_sym_snippets


if __name__ == '__main__':
    # dloader = dl.DataModule(dir='', max_raw_length=200, max_event_length=30, bases_offset=2, batch_size=64, train_size=1, val_size=0, test_size=0,
    #                         load_source='chiron', event_detection=True)

    # start = timer()
    # ds = dloader.get_dataset_from_single_chiron('data/chiron/train/ecoli_0001_0002/ecoli_0001.signal')
    # print(timer() - start)
    raw = np.loadtxt('data/chiron/train/ecoli_0001_0002/ecoli_0001.signal', dtype=int)
    label = np.loadtxt('data/chiron/train/ecoli_0001_0002/ecoli_0001.label', dtype=object)
    nuc_raw_ranges = label[:, :2].astype(int)
    labels = label[:,2]


    prepare_snippets(raw, nuc_raw_ranges, labels, 2)
