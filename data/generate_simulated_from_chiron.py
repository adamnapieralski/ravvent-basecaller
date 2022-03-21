import numpy as np
from pathlib import Path
import subprocess
import shlex

def get_bases_sequence_from_label_file(label_path):
    labels = np.loadtxt(label_path, dtype='object')
    bases_sequence = ''.join(labels[:,2])
    return bases_sequence

def create_fasta_file(path, sequence):
    with open(path, 'wt') as f:
        f.write('>0\n{}'.format(sequence))

def convert_label_to_fasta(label_path, fasta_dir_path):
    sequence = get_bases_sequence_from_label_file(label_path)
    file_name = label_path.stem
    fasta_path = fasta_dir_path / (file_name + '.fasta')
    create_fasta_file(fasta_path, sequence)

def convert_all_dir_labels_to_fastas(label_dir_path, fasta_dir_path):
    label_dir_path = Path(label_dir_path)
    labels_files = [p for p in label_dir_path.iterdir() if p.suffix == '.label']
    labels_files.sort()

    for lfp in labels_files:
        convert_label_to_fasta(lfp, Path(fasta_dir_path))

def run_deepsimulator(data_path, seq_name, random_seed=0):
    cmd = 'docker run -it --rm -v {}:/deepsimulator/data/ -e UID=1000 adamnapieralski/deepsimulator -i data/fasta/{}.fasta -o data/{} -G 1 -B 2 -n -1 -P 0 -O 1 -S {} -M 0'.format(data_path, seq_name, seq_name, random_seed)
    subprocess.run(shlex.split(cmd))

def run_deepimulator_all_fasta(data_path):
    data_path = Path(data_path)
    fastas_path = data_path / 'fasta'
    seq_names = [p.stem for p in fastas_path.iterdir() if p.suffix == '.fasta']
    seq_names.sort()

    for i, seq_name in enumerate(seq_names):
        print(str(data_path), seq_name, i)
        run_deepsimulator(str(data_path), seq_name, i)

def convert_simulator_signal_to_signal(sim_out_dir, signal_path):
    vals = np.loadtxt(sim_out_dir / 'signal/signal_0.txt', dtype=int)
    with open(signal_path, 'wt') as f:
        f.write('500 ' + ' '.join([str(v) for v in vals]))


def prepare_event_detection(sim_out_dir, event_detection_path):
    fast5_path = next((sim_out_dir / "fast5").iterdir())
    cmd = '/home/ubuntu/event_detection/detect_events --win-len1 5 --win-len2 13 {}'.format(str(fast5_path))
    with open(event_detection_path, 'wt') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

def prepare_label(sim_out_dir, label_path):
    sequence = ""
    with open(sim_out_dir / 'sampled_read.fasta', 'rt') as f:
        sequence = f.readlines()[1].rstrip()
    align_path = next((sim_out_dir / 'align').iterdir())
    align = np.loadtxt(align_path, dtype=int)
    with open(label_path, 'wt') as f:
        for i in range(len(sequence)):
            w = np.where(align[:,1] == i+1)
            start_id = w[0][0] + 1
            end_id = w[0][-1] + 2
            f.write('{} {} {}\n'.format(start_id, end_id, sequence[i]))

def convert_sim_out_to_chiron(sim_out_dir, seq_name, out_dir):
    sim_out_dir = Path(sim_out_dir)
    out_dir = Path(out_dir)
    convert_simulator_signal_to_signal(sim_out_dir, out_dir / '{}.signal'.format(seq_name))
    prepare_event_detection(sim_out_dir, out_dir / '{}.eventdetection'.format(seq_name))
    prepare_label(sim_out_dir, out_dir / '{}.label'.format(seq_name))

def convert_multi_sim_out_to_chiron(sims_out_dir, out_dir):
    sims_out_dir = Path(sims_out_dir)
    sims_out_dirs = [d for d in sims_out_dir.iterdir() if 'Lambda' in str(d)]
    sims_out_dirs.sort()
    for s_dir in sims_out_dirs[:80]:
        print(s_dir)
        convert_sim_out_to_chiron(str(s_dir), s_dir.stem, out_dir)

if __name__ == '__main__':
    # convert_all_dir_labels_to_fastas('/home/ubuntu/ravvent-basecaller/data/chiron/lambda/eval/all', '/home/ubuntu/ravvent-basecaller/data/chiron/lambda/eval/simulator/fasta')
    # run_deepimulator_all_fasta('/home/ubuntu/ravvent-basecaller/data/chiron/lambda/eval/simulator')
    # convert_sim_out_to_chiron('/home/ubuntu/ravvent-basecaller/data/chiron/lambda/train/simulator/Lambda_0001', 'Lambda_sim_0001', '/home/ubuntu/ravvent-basecaller/out_dir')
    convert_multi_sim_out_to_chiron('/home/ubuntu/ravvent-basecaller/data/chiron/lambda/eval/simulator', '/home/ubuntu/ravvent-basecaller/data/chiron/lambda/eval/all_sim')