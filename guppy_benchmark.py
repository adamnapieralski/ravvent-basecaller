import h5py
import shutil
import subprocess, shlex
import numpy as np
from pathlib import Path
from data_loader import DataModule
import re

raw_max_len = 200
event_max_len = 30
batch_size = 128
bases_offset = 8

RANDOM_SEED = 22


def prepare_boilerplate_fast5_file(reference_fast5_path, boilerplate_file):
    shutil.copyfile(reference_fast5_path, boilerplate_file)
    ref_file = h5py.File(boilerplate_file, 'r+')
    raw_dat = list(ref_file['/Raw/Reads/'].values())[0]
    del raw_dat['Signal']
    ref_file.close()

def create_new_data_bit(id, raw_values, bases, boilerplate_file, fast5_dir, fasta_dir):
    fast5_path = f'{fast5_dir}/seq_{id}.fast5'
    shutil.copyfile(boilerplate_file, fast5_path)
    file = h5py.File(fast5_path, 'r+')
    raw_dat = list(file['/Raw/Reads/'].values())[0]
    raw_attrs = raw_dat.attrs
    raw_dat.create_dataset('Signal',data=raw_values, dtype='i2', compression='gzip', compression_opts=9)  #-> with compression
    raw_attrs['duration'] = raw_values.size
    raw_attrs['read_id'] = f'seq_{id}'
    file.close()

    fasta_path = f'{fasta_dir}/seq_{id}.fasta'
    with open(fasta_path, 'wt') as f:
        f.write(">{}\n".format(id))
        f.write(bases)

def run_guppy_basecalling(fast5_dir, fastq_dir, batch_size=10000):
    cmd = 'guppy_basecaller -i {} -s {} -c dna_r9.4.1_450bps_hac.cfg --read_batch_size {}'.format(fast5_dir, fastq_dir, batch_size)
    subprocess.run(shlex.split(cmd))

def get_fast5_fasta_fastq_mapping(fastq_dir):
    summary = np.loadtxt(f'{fastq_dir}/sequencing_summary.txt', skiprows=1, dtype='object')
    summary = summary[:, 0:3]
    for row in summary:
        row[1] = row[0].replace('fast5', 'fasta')
        row[2] = f'fastq_runid_{row[1]}_0_0.fastq'
    return summary

def get_bases_sequences(fastq_dir, fasta_dir):
    fastq_p = Path(fastq_dir)
    fasta_p = Path(fasta_dir)

    bases_sequence = []
    for fastq_file in [f for f in fastq_p.iterdir() if '.fastq' in str(f)]:
        seq_id = ''
        ref_loaded = False
        with open(fastq_file, 'rt') as f:
            for line in f:
                if line.startswith('@'):
                    res = re.match(r'\@(seq_\d+).*', line)
                    seq_id = res.group(1)
                    with open(fasta_p / f'{seq_id}.fasta', 'rt') as fa:
                        bases_ref = fa.readlines()[1]
                        ref_loaded = True
                elif ref_loaded:
                    bases_sequence.append((bases_ref, line.strip()))
                    ref_loaded = False
    return np.array(bases_sequence)

def single_bases_accuracy(ref, pred):
    match = 0
    for i, base in enumerate(ref):
        if i >= len(pred):
            break
        if base == pred[i]:
            match += 1
    return match / len(pred)



def load_data(data_string):
    dm = DataModule(
        dir='data/simulator/reduced/{}.eval'.format(data_string),
        max_raw_length=raw_max_len,
        max_event_length=event_max_len,
        bases_offset=bases_offset,
        batch_size=batch_size,
        train_size=0,
        val_size=0.25,
        test_size=0.75,
        load_source='simulator',
        event_detection=True,
        random_seed=RANDOM_SEED,
        verbose=True
    )

    raw_aligned_data, events_sequence, bases_sequence, alignment_data = dm._load_simulator_data(dm.dir, event_detection=dm.event_detection)
    raw_samples, event_samples, bases_samples = zip(*dm.samples_generator(raw_aligned_data, events_sequence, bases_sequence, dm.max_raw_length, dm.bases_offset, alignment_data, dm.event_detection))
    raw_data, event_data = dm.pad_input_data(raw_samples, event_samples)
    # raw_data_split, event_data_split, bases_data_split = dm.train_val_test_split(raw_data, event_data, bases_samples, train_size=dm.train_size, val_size=dm.val_size, test_size=dm.test_size)
    raw_data_split, event_data_split, bases_data_split = raw_data, event_data, bases_samples

    # joined_bases_data_split = []
    # for bases_data in bases_data_split:
    #     if bases_data is None:
    #         joined_bases_data_split.append(None)
    #         continue
    #     joined_bases_data_split.append([''.join(bases_sample) for bases_sample in bases_data])
    bases_data_split = [''.join(bases_sample) for bases_sample in bases_data_split]

    # return raw_data_split[2], joined_bases_data_split[2], dm.input_padding_value
    return raw_data_split, bases_data_split, dm.input_padding_value

def generate_guppy_data(raw_data, bases_data, padding_value, boilerplate_file, fast5_dir, fasta_dir, fastq_dir):
    for id, (raw_seq, bases_seq) in enumerate(zip(raw_data, bases_data)):
        create_new_data_bit(id, raw_seq[raw_seq != padding_value], bases_seq, boilerplate_file, fast5_dir, fasta_dir)

    run_guppy_basecalling(fast5_dir, fastq_dir, len(bases_data))

if __name__ == '__main__':
    data_string = 'seq.3.10000.45'
    guppy_seq_dir = f'guppy_benchmark/{data_string}'
    boilerplate_file = 'guppy_benchmark/boilerplate.fast'
    prepare_boilerplate_fast5_file('guppy_benchmark/reference.fast5', boilerplate_file)

    fast5_dir, fasta_dir, fastq_dir = f'{guppy_seq_dir}/fast5', f'{guppy_seq_dir}/fasta', f'{guppy_seq_dir}/fastq'

    raw_data, bases_data, padding_value = load_data(data_string)

    generate_guppy_data(raw_data, bases_data, padding_value, boilerplate_file, fast5_dir, fasta_dir, fastq_dir)