import numpy as np
import data_loader as dl

if __name__ == '__main__':
    total_lens = np.array([], dtype=np.int8)

    for i, data_file_num in enumerate([f'{i:04d}' for i in range(1, 2001)]):
        print(f'ecoli_{data_file_num}')
        signal_path = f'data/chiron/ecoli/train/ecoli_{data_file_num}.signal'
        label_path = f'data/chiron/ecoli/train/ecoli_{data_file_num}.label'
        raw = np.loadtxt(signal_path, dtype=int)
        label = np.loadtxt(label_path, dtype=object)
        nuc_raw_ranges = label[:, :2].astype(int)
        nuc_reference_symbols = label[:,2]

        raw_snippets, event_snippets, nuc_sym_snippets = dl.prepare_snippets(raw, nuc_raw_ranges, nuc_reference_symbols, 1)
        lens = []
        for ev_snippet in event_snippets:
            lens.append(len(ev_snippet))
        total_lens = np.append(total_lens, lens)

        if i % 100 == 0:
            np.save('ecoli_event_counts.npy', total_lens)

    np.save('ecoli_event_counts.npy', total_lens)



    total_lens = np.array([], dtype=np.int8)

    for i, data_file_num in enumerate([f'{i:04d}' for i in range(1, 2001)]):
        print(f'Lambda_{data_file_num}')
        signal_path = f'data/chiron/lambda/train/all/Lambda_{data_file_num}.signal'
        label_path = f'data/chiron/lambda/train/all/Lambda_{data_file_num}.label'
        raw = np.loadtxt(signal_path, dtype=int)
        label = np.loadtxt(label_path, dtype=object)
        nuc_raw_ranges = label[:, :2].astype(int)
        nuc_reference_symbols = label[:,2]

        raw_snippets, event_snippets, nuc_sym_snippets = dl.prepare_snippets(raw, nuc_raw_ranges, nuc_reference_symbols, 1)
        lens = []
        for ev_snippet in event_snippets:
            lens.append(len(ev_snippet))
        total_lens = np.append(total_lens, lens)

        if i % 100 == 0:
            np.save('Lambda_event_counts.npy', total_lens)

    np.save('Lambda_event_counts.npy', total_lens)


    for sim_i in [3, 12, 21, 43, 4096]:
        total_lens = np.array([], dtype=np.int8)
        signal_path = f'data/chiron/lambda/train/all/Lambda_{data_file_num}.signal'
        label_path = f'data/chiron/lambda/train/all/Lambda_{data_file_num}.label'
        raw = np.loadtxt(signal_path, dtype=int)
        label = np.loadtxt(label_path, dtype=object)
        nuc_raw_ranges = label[:, :2].astype(int)
        nuc_reference_symbols = label[:,2]

        raw_snippets, event_snippets, nuc_sym_snippets = dl.prepare_snippets(raw, nuc_raw_ranges, nuc_reference_symbols, 1)
        lens = []
        for ev_snippet in event_snippets:
            lens.append(len(ev_snippet))
        total_lens = np.append(total_lens, lens)

        np.save('Lambda_event_counts.npy', total_lens)