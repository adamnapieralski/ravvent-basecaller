import numpy as np
import tensorflow as tf
import subprocess
import shlex
import os
from pathlib import Path

from Bio import pairwise2
import utils
import merger
import data_loader_2 as dl
import merger
import basecaller_s2s as bc_s2s
import basecaller_s2s_2 as bc_s2s_2
import json

class MappingEvaluator():

    def __init__(self, load_source='chiron'):
        self.merger = merger.Merger()
        self.dm = dl.DataModule(
            dir='', max_raw_length=200, max_event_length=30, bases_offset=6, batch_size=128,
            train_size=1, val_size=0, test_size=0, load_source=load_source, event_detection=True
        )
        self.basecaller = None

    def run(self, signal_data_source):
        label_path = Path(signal_data_source).with_suffix('.label')
        ref_seq = ''.join(list(np.loadtxt(label_path, dtype='object')[:,2]))

        dataset = self.dm.get_dataset_from_single_chiron(signal_data_source, batched=False)
        dataset = dataset.batch(len(dataset))
        for data in dataset:
            break

        input_data, target_data = utils.unpack_data_to_input_target(data, self.basecaller.input_data_type)

        pred_tokens, beam_scores = self.basecaller.beam_search_prediction(input_data, beam_width=5)
        scores = utils.calc_prob_logits_beam_search_scores(beam_scores).numpy()

        seqs = self.basecaller.tokens_to_bases_sequence(tf.cast(pred_tokens, tf.int64))
        seqs = [s.numpy().decode('UTF-8') for s in seqs]

        nuc_preds = [
            merger.SeqLogitsPair(seq, list(sc[:len(seq)])) for seq, sc in zip(seqs, scores)
        ]

        merged_seq = self.merger.merge(nuc_preds).seq

        fasta_path = 'temp/ref.fasta'
        fastq_path = 'temp/pred.fastq'
        mapping_path = 'temp/mapping.paf'
        self._create_fasta(ref_seq, fasta_path)
        self._create_fastq(merged_seq, fastq_path)
        self._run_minimap(fasta_path, fastq_path, mapping_path)

        ident = self._read_mapping_identity(mapping_path)

        # os.remove(fasta_path)
        # os.remove(fastq_path)
        # os.remove(mapping_path)

        return ident

    def _create_fasta(self, seq, fname):
        with open(fname, 'wt') as f:
            f.write(f'>{seq[:10]}\n{seq}')

    def _create_fastq(self, seq, fname):
        with open(fname, 'wt') as f:
            f.write(f'@{seq[:10]}\n')
            f.write(seq + '\n')
            f.write('+\n')
            f.write('!' * len(seq))

    def _run_minimap(self, ref_path, pred_path, out_path):
        cmd = f'minimap2 -x map-ont -c {ref_path} {pred_path}'
        with open(out_path, 'wt') as f:
            subprocess.run(shlex.split(cmd), stdout=f)

    def _read_mapping_identity(self, mapping_path):
        matches = 0
        total_blocks_len = 0
        with open(mapping_path, 'rt') as paf:
            for line in paf:
                paf_parts = line.strip().split('\t')
                if len(paf_parts) < 11:
                    continue
                matches += int(paf_parts[9])
                total_blocks_len += int(paf_parts[10])
        if total_blocks_len == 0:
            return 0
        return matches / total_blocks_len

    def setup_basecaller(self, weights_path, mode=1):
        if mode == 1:
            self.basecaller = bc_s2s.Basecaller(
                units=128,
                batch_sz=128,
                output_text_processor=self.dm.output_text_processor,
                input_data_type='raw',
                input_padding_value=self.dm.initial_input_padding_value,
                rnn_type='bilstm',
                teacher_forcing=0.5,
                attention_type='luong',
                beam_width=5
            )
        elif mode == 2:
            self.basecaller = bc_s2s_2.Basecaller(
                enc_units=128, dec_units=96, batch_sz=128, output_text_processor=self.dm.output_text_processor,
                input_data_type='raw', input_padding_value=self.dm.input_padding_value,
                encoder_depth=3, decoder_depth=3, rnn_type='bilstm', attention_type='luong',
                teacher_forcing=0.5
            )


        self.basecaller.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.0001, clipnorm=1),
        )
        self.basecaller.load_weights(weights_path)

if __name__ == '__main__':
    me = MappingEvaluator(load_source='tayiaki')
    me.setup_basecaller(
        'models/taiyaki/model.s2s.raw.lr0.0001.bilstm.encu128.encd3.decu96.decd3.chiron.rawmax200.b128.ep99.pat99.embone_hot.ed1.luong.tf0.5.do02.boff6.08/model_chp',
        mode=2
    )

    with open('data/taiyaki/samples.rawmax200.evmax30.offset6/files_info_val.json', 'rt') as f:
        val_files = json.load(f)
        val_files = [v['path'].replace('/samples.rawmax200.evmax30.offset6', '').replace('pkl', 'signal') for v in val_files]

    for v in val_files:
        print(f'Running {v}')
        ident = me.run(v)
        print(ident)