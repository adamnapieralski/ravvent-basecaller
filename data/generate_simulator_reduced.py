import random
import sys
import subprocess
import shlex
import itertools
import json
from typing import List

bases = ['A', 'C', 'G', 'T']
k_num = 6
random_seed = 22

initial_mers = [
    'TGCACT', 'ATCGGT', 'GACCTA', 'TCTTAC', 'ATAAAG', 'AGTCTT', 'TCGGCG', 'GACACT', 'GACCCA', 'CCACCC'
]

mers_all = [m for m in itertools.product(bases, repeat=k_num)]

# random.seed(random_seed)

def get_random_kmer():
    return ''.join(random.choice(mers_all))

def get_kmer_list(n: int):
    if n <= len(initial_mers):
        return initial_mers[0:n]
    else:
        mers = initial_mers
        for _ in range(n - len(initial_mers)):
            new_mer = get_random_kmer()
            while new_mer in mers:
                new_mer = get_random_kmer()
            mers.append(new_mer)
        return mers

def compute_number_of_appearing_mers(kmers_list):
    def compute_rec(passed, remaining, appearing_set, appearing_nums_acc):
        if len(remaining) == 0:
            return appearing_nums_acc
        new_mer = remaining.pop(0)

        checked = new_mer + new_mer

        for j in range(len(new_mer)):
            appearing_set.add(checked[j:j+len(new_mer)])

        for i in range(len(passed)):
            checked_comb = [passed[i] + new_mer, new_mer + passed[i]]
            for checked in checked_comb:
                for j in range(len(new_mer) - 1):
                    appearing_set.add(checked[j+1:j+1+len(new_mer)])

        appearing_nums_acc.append(len(appearing_set))
        passed.append(new_mer)
        return compute_rec(passed, remaining, appearing_set, appearing_nums_acc)

    appearing_nums_acc = compute_rec([], kmers_list, set(), [])
    return appearing_nums_acc[-1], appearing_nums_acc

def generate_random_sequence_from_k_mer_list(length: int, k_mers: List[str]):
    seq = ''
    for _ in range(length // (len(k_mers[0]))):
        seq += random.choice(k_mers)
    if len(seq) < length:
        seq += random.choice(k_mers)[0:length - len(seq)]
    return seq

def load_mers():
    mers = []
    with open('simulator-6-mers.json', 'rt') as f:
        mers = json.load(f)
    return mers

def run_deepsimulator(fasta_path, output_path, random_seed=0):
    cmd = '/home/napiad/Programs/DeepSimulator/deep_simulator.sh -H /home/napiad/Programs/DeepSimulator/ -i {} -o {} -G 1 -B 2 -n -1 -P 0 -O 1 -S {} -M 0'.format(fasta_path, output_path, random_seed)
    subprocess.run(shlex.split(cmd))

if __name__ == '__main__':
    mers = load_mers()

    generation_details = [
        {'id': 3, 'length': 10000, '6_mers_num': 45}, # 45 6-mers
        {'id': 12, 'length': 25000, '6_mers_num': 450}, # 450 6-mers
        {'id': 21, 'length': 50000, '6_mers_num': 1024}, # 1024 6-mers
        {'id': 43, 'length': 100000, '6_mers_num': 2048}, # 2048 6-mers
        {'id': 4096, 'length': 200000, '6_mers_num': 4096} # all
    ]

    for det in generation_details:
        for purpose_type in ['train', 'eval']:
            filename = 'seq.{}.{}.{}.{}.fasta'.format(det['id'], det['length'], det['6_mers_num'], purpose_type)
            seq = generate_random_sequence_from_k_mer_list(det['length'], mers[0:det['id']])
            with open(filename, 'wt') as f:
                f.write(">{}\n".format(filename.replace('.fasta', '')))
                f.write(seq)
            run_deepsimulator(filename, filename.replace('.fasta', ''), random_seed=det['id'] if purpose_type == 'train' else 2 * det['id'])
