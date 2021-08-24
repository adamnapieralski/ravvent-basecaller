import subprocess
import shlex
import os

def run_merger(a_seq_file, b_seq_file, out_seq_file, align_file):
    cmd = f'merger -asequence {a_seq_file} -bsequence {b_seq_file} -outseq {out_seq_file} -osformat4 text -outfile {align_file} -datafile EDNAFULL -auto'
    subprocess.run(shlex.split(cmd))

def create_seq_file(path, sequence):
    with open(path, 'w') as f:
        f.write(sequence)

def read_seq_from_file(path):
    with open(path, 'r') as f:
        return f.readline()

def prepend_to_seq_file(path, pre_sequence):
    seq = read_seq_from_file(path)
    create_seq_file(path, pre_sequence + seq)

def merge_chunks(chunks):
    a_seq_file = 'a.seq'
    b_seq_file = 'b.seq'
    align_file = 'align.merger'
    merged_file = 'merged.seq'
    create_seq_file('a.seq', chunks[0])
    create_seq_file('b.seq', chunks[1])
    run_merger(a_seq_file, b_seq_file, merged_file, align_file)
    for i, chunk in enumerate(chunks[2:]):
        merged_seq = read_seq_from_file(merged_file)
        suffix_from_merged = merged_seq[(len(merged_seq) - len(chunks[i-2])):]
        const_from_merged = merged_seq[0:-len(chunks[i-2])]
        create_seq_file(a_seq_file, suffix_from_merged)
        create_seq_file(b_seq_file, chunk)
        run_merger(a_seq_file, b_seq_file, merged_file, align_file)
        prepend_to_seq_file(merged_file, const_from_merged)

    os.remove(a_seq_file)
    os.remove(b_seq_file)
    os.remove(align_file)
    seq = read_seq_from_file(merged_file)
    os.remove(merged_file)
    return seq
