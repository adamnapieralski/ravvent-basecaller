from pathlib import Path
import shlex
import subprocess
import os
import json
from timeit import default_timer as timer
import re


def read_mapping_identity(mapping_path):
    matches = 0
    total_blocks_len = 0
    read_length = 0
    with open(mapping_path, 'rt') as paf:
        for line in paf:
            paf_parts = line.strip().split('\t')
            if len(paf_parts) < 11:
                continue
            read_length = int(paf_parts[1])
            matches += int(paf_parts[9])
            total_blocks_len += int(paf_parts[10])

    return {
        'read_length': read_length,
        'matches': matches,
        'total_block_len': total_blocks_len,
        'identity': matches / total_blocks_len if total_blocks_len != 0 else 0.
    }

def run_guppy_single_dir(dir_path, mode='cpu'):
    out_dir = dir_path / mode
    os.mkdir(out_dir)

    if mode == 'cpu':
        cmd = f'/home/ubuntu/ont-guppy/bin/guppy_basecaller --input_path {str(dir_path)} --save_path {str(out_dir)} -c dna_r9.4.1_450bps_hac.cfg'
    elif mode == 'gpu':
        cmd = f'/home/ubuntu/ont-guppy/bin/guppy_basecaller --input_path {str(dir_path)} --save_path {str(out_dir)} -c dna_r9.4.1_450bps_hac.cfg -x auto'

    start = timer()
    subprocess.run(shlex.split(cmd), stdout=subprocess.DEVNULL)
    elapsed = timer() - start

    fastq_path = [v for v in out_dir.iterdir() if '.fastq' == v.suffix][0]
    fasta_path = [v for v in dir_path.iterdir() if '.fasta' == v.suffix][0]

    mapping_path = out_dir / 'mapping.paf'
    cmd = f'minimap2 -x map-ont -c {str(fasta_path)} {str(fastq_path)}'
    with open(mapping_path, 'wt') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = read_mapping_identity(mapping_path)
    res['time_measured'] = elapsed

    log_path = [v for v in out_dir.iterdir() if '.log' == v.suffix][0]
    with open(log_path, 'rt') as f:
        logs = ' '.join(f.readlines())

    init_match = re.search(r'Init time: (\d+) ', logs)
    caller_match = re.search(r'.*Caller time: (\d+) ms, Samples called: (\d+)', logs)

    init_time = 0
    caller_time = 0
    samples_called = 0
    if len(init_match.groups()) == 1 and len(caller_match.groups()) == 2:
        init_time = int(init_match[1])
        caller_time = int(caller_match[1])
        samples_called = int(caller_match[2])

    res['init_time'] = init_time / 1000
    res['caller_time'] = caller_time / 1000
    res['samples_called'] = samples_called

    return res

def run_guppy_all_dirs(general_dir, res_path, mode='cpu'):
    general_dir = Path(general_dir)
    all_dirs = [d for d in general_dir.iterdir() if d.is_dir()]
    all_dirs.sort()
    results = []
    for d in all_dirs:
        res = run_guppy_single_dir(d, mode=mode)
        results.append(res)

        with open(res_path, 'wt') as f:
            json.dump(results, f, indent=2)

def calculate_speed(res_path):
    with open(res_path, 'rt') as f:
        results = json.load(f)

    pred_bases, init_time, caller_time, samples_called, own_time = 0, 0, 0, 0, 0
    for res in results:
        pred_bases += res['read_length']
        init_time += res['init_time']
        caller_time +=res['caller_time']
        samples_called += res['samples_called']
        own_time += res['time_measured']

    total_time = init_time + caller_time
    print(pred_bases / caller_time, samples_called / caller_time, pred_bases / total_time, samples_called / total_time)

if __name__ == '__main__':
    mode = 'cpu'
    run_guppy_all_dirs('data/chiron/lambda/lambda_guppy_val', f'info/snippets/perform.guppy.{mode}.json', mode=mode)

    mode = 'gpu'
    run_guppy_all_dirs('data/chiron/lambda/lambda_guppy_val', f'info/snippets/perform.guppy.{mode}.json', mode=mode)

