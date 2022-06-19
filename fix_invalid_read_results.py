import time
import numpy as np
import tensorflow as tf
import subprocess
import shlex
import os
from pathlib import Path

import utils
import data_loader as dl
import merger_smooth
import basecaller_multi as bc_multi
import basecaller_old as bc_old

import json

BEAM_WIDTH = 1
ENCODER_DEPTH = 3
DECODER_DEPTH = 2

class MappingEvaluator():

    def __init__(self, merger_scores_id=0):
        self.merger = merger_smooth.Merger(scores_id=merger_scores_id)
        self.stride = 6
        self.basecaller = None

    def _split_into_chunks(self, arr, def_chunk_size):
        """Splits array into chunks of def_chunk_size, the last chunk is left with the size <= def_chunk_size
        """
        return np.array_split(arr, np.arange(1, arr.shape[0] // def_chunk_size + 1) * def_chunk_size)

    def run(self, signal_data_source, chunk_size=1024):
        label_path = Path(signal_data_source).with_suffix('.label')
        ref_seq = ''.join(list(np.loadtxt(label_path, dtype='object')[:,2]))

        raw_snippets, event_snippets, nuc_tk_snippets = dl.load_data_from_single_signal_label(signal_data_source, label_path, self.stride)

        raw_snippets_chunked = self._split_into_chunks(raw_snippets, chunk_size)
        event_snippets_chunked = self._split_into_chunks(event_snippets, chunk_size)
        nuc_tk_snippets_chunked = self._split_into_chunks(nuc_tk_snippets, chunk_size)

        nuc_preds = []

        for raw_snippets, event_snippets, nuc_tk_snippets in zip(raw_snippets_chunked, event_snippets_chunked, nuc_tk_snippets_chunked):
            data = (
                tf.convert_to_tensor(raw_snippets), tf.convert_to_tensor(event_snippets), tf.convert_to_tensor(nuc_tk_snippets))

            input_data, target_data = utils.unpack_data_to_input_target(data, self.basecaller.input_data_type)

            pred_tokens, beam_scores = self.basecaller.beam_search_prediction(input_data, beam_width=BEAM_WIDTH, max_output_len=tf.shape(target_data)[1])
            scores = utils.calc_prob_logits_beam_search_scores(beam_scores).numpy()

            seqs = self.basecaller.tokens_to_nuc_sequences(pred_tokens)

            nuc_preds.extend([
                merger_smooth.SeqLogitsPair(seq, list(sc[:len(seq)])) for seq, sc in zip(seqs, scores)])

        print([s.seq for s in nuc_preds])
        return {}

        # merged_seq = self.merger.merge(nuc_preds).seq

        # fasta_path = 'temp_fix/ref.fasta'
        # fastq_path = 'temp_fix/pred.fastq'
        # mapping_path = 'temp_fix/mapping.paf'
        # self._create_fasta(ref_seq, fasta_path)
        # self._create_fastq(merged_seq, fastq_path)
        # self._run_minimap(fasta_path, fastq_path, mapping_path)

        # ident = self._read_mapping_identity(mapping_path)

        # os.remove(fasta_path)
        # os.remove(fastq_path)
        # os.remove(mapping_path)

        # return ident

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

    def setup_basecaller(self, weights_path, data_type, mode=1):
        if mode == "old":
            self.basecaller = bc_old.Basecaller(
                enc_units=128,
                dec_units=128,
                batch_sz=128,
                tokenizer=dl.nuc_tk,
                input_data_type=data_type,
                input_padding_value=dl.INPUT_PADDING,
                encoder_depth=ENCODER_DEPTH,
                decoder_depth=DECODER_DEPTH,
                rnn_type='bilstm',
                attention_type='luong',
                teacher_forcing=0.5
            )
        elif mode == "multi":
            self.basecaller = bc_multi.Basecaller(
                enc_units=128,
                dec_units=128,
                batch_sz=128,
                tokenizer=dl.nuc_tk,
                input_data_type=data_type,
                input_padding_value=dl.INPUT_PADDING,
                encoder_depth=ENCODER_DEPTH,
                decoder_depth=DECODER_DEPTH,
                rnn_type='bilstm',
                attention_type='luong',
                teacher_forcing=0.5
            )
        self.basecaller.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.0001, clipnorm=1.),
        )
        self.basecaller.load_weights(weights_path)

    def compute_total_results(self, results_path):
        results = []
        matches, total_block_len, invalid_ref_len = 0, 0, 0
        invalid = 0
        with open(results_path, 'rt') as f:
            results = json.load(f)

        for res in results:
            matches += res['matches']
            total_block_len += res['total_block_len']
            if res['read_length'] == 0:
                invalid += 1
                invalid_ref_len += res['ref_length']

        wx_total = 0
        w_total = 0

        wx_valid = 0
        w_valid = 0

        invalid_num = 0

        for res in results:
            identity = 0
            if res['read_length'] != 0:
                identity = res['matches'] / res['total_block_len']
                wx_valid += identity * res['ref_length']
                w_valid += res['ref_length']
            else:
                invalid_num += 1
            wx_total += identity * res['ref_length']
            w_total += res['ref_length']

        identity_score_total = wx_total / w_total * 100 if w_valid > 0 else 0
        identity_score_valid = wx_valid / w_valid * 100 if w_valid > 0 else 0

        invalid_frac = invalid_num / len(results) * 100

        return round(identity_score_total, 3), round(identity_score_valid, 3), round(invalid_frac, 3)

        match_score = matches / total_block_len if total_block_len != 0 else 0.
        match_score_with_invalid = matches / (total_block_len + invalid_ref_len)
        invalid_fraction = invalid / len(results)

        return match_score, match_score_with_invalid, invalid_fraction

    def analyse_and_select_best_results(self, results_dir, data_type):
        results_dir = Path(results_dir)
        results_paths = [p for p in results_dir.iterdir() if data_type in str(p)]
        results_paths.sort()

        scores = []
        for res_path in results_paths:
            match_score, match_score_with_invalid, invalid_fraction = self.compute_total_results(res_path)
            scores.append(match_score_with_invalid)
            print(res_path.stem)
            print(match_score, match_score_with_invalid, invalid_fraction)

        print(f'Best score: {np.max(scores)} of {results_paths[np.argmax(scores)].stem}')

    def add_ref_length_to_results(self, results_path):
        results = []
        with open(results_path, 'rt') as f:
            results = json.load(f)

        if len(results) > 0:
            for res in results:
                label_path = res['path'].replace('.signal', '.label')
                labels = np.loadtxt(label_path, dtype=object)
                res['ref_length'] = labels.shape[0]
            with open(results_path, 'wt') as f:
                json.dump(results, f, indent=2)

    def evaluate_specific(self, data_type, ep_start, ep_end, eval_type='val'):
        for ep in [f'{i:02d}' for i in range(ep_start, ep_end + 1, 1)]:
            res = []

            self.setup_basecaller(
                f'models/snippets/mask/encd_{ENCODER_DEPTH}_decd_{DECODER_DEPTH}/model.1.{data_type}.lambda.mask.pad.lr0.0001.bilstm.encu128.encd{ENCODER_DEPTH}.decu128.decd{DECODER_DEPTH}.b128.luong.tf0.5.strd6.spe10000.spv1500.{ep}/model_chp',
                data_type,
                mode=1
            )
            if eval_type == 'val':
                files_info_path = 'data/chiron/lambda/eval/all/files_info.val.snippets.stride_6.json'
                eval_res_path = f'info/snippets/mapping_evaluations/encd_{ENCODER_DEPTH}_decd_{DECODER_DEPTH}/mapping_evaluator_results.snippets.{data_type}.{ep}.json'
            elif eval_type == 'test':
                # files_info_path = 'data/chiron/lambda/eval/all/files_info.test.snippets.stride_6.json'
                # eval_res_path = f'info/snippets/mapping_evaluations/mapping_evaluator_results.snippets.test.{data_type}.{ep}.beam{BEAM_WIDTH}.json'
                files_info_path = 'data/chiron/ecoli/eval/files_info.test.snippets.stride_6.json'
                eval_res_path = f'info/snippets/mapping_evaluations/encd_{ENCODER_DEPTH}_decd_{DECODER_DEPTH}/test/mapping_evaluator_results.snippets.test.{data_type}.{ep}.ecoli.beam{BEAM_WIDTH}.json'

            with open(files_info_path, 'rt') as f:
            # with open('data/chiron/ecoli/eval/files_info.val.snippets.stride_6.json', 'rt') as f:
                val_files = json.load(f)
                # done_paths = [v['path'] for v in res]
                val_files = [v['signal_path'] for v in val_files] # if v['signal_path'] not in done_paths]

            for v in val_files:
                print(f'Running {v}')
                ident_read = self.run(v)
                ident_read['path'] = v
                ident_read['ref_length'] = np.loadtxt(v.replace('.signal', '.label'), dtype=object).shape[0]
                print(ident_read)
                res.append(ident_read)
                with open(eval_res_path, 'wt') as f:
                    json.dump(res, f, indent=2)


    def fix_results(self, results_path):
        if 'beam1' in str(results_path):
            BEAM_WIDTH = 1
        else:
            BEAM_WIDTH = 5

        if 'fixed' in str(results_path):
            return

        fixed_results_path = results_path.parent / f"{results_path.stem}.fixed.json"

        with open(results_path, 'rt') as f:
            old_results = json.load(f)

        fixed_results = old_results

        invalid_count = len([r for r in old_results if r['read_length'] == 0])
        counter = 0

        print(f"Fixing results: {str(results_path)}\nInvalid count: {invalid_count}\n")

        for i, old_res in enumerate(old_results):
            if old_res['read_length'] == 0:
                counter += 1
                ident_read = self.run(old_res['path'])
                ident_read['path'] = old_res['path']
                ident_read['ref_length'] = np.loadtxt(old_res['path'].replace('.signal', '.label'), dtype=object).shape[0]
                print(ident_read)
                fixed_results[i] = ident_read

                # with open(fixed_results_path, 'wt') as f:
                #     json.dump(fixed_results, f, indent=2)

                print(f"{counter} / {invalid_count} ({str(results_path)})")

                if counter >= 1:
                    break



test_results_meta = {
    (3, 2): {
        "raw": {
            "id": 36,
            "basecaller": "multi",
        },
        # "joint": {
        #     "id": 40,
        #     "basecaller": "multi",
        # },
        # "event": {
        #     "id": 39,
        #     "basecaller": "multi",
        # },
    },
    # (3, 1): {
    #     "raw": {
    #         "id": 37,
    #         "basecaller": "multi",
    #     },
    #     "joint": {
    #         "id": 40,
    #         "basecaller": "multi",
    #     },
    #     "event": {
    #         "id": 38,
    #         "basecaller": "multi",
    #     },
    # },
    # (2, 2): {
    #     "raw": {
    #         "id": 36,
    #         "basecaller": "old",
    #     },
    #     "joint": {
    #         "id": 40,
    #         "basecaller": "old",
    #     },
    #     "event": {
    #         "id": 39,
    #         "basecaller": "old",
    #     },
    # },
    # (2, 1): {
    #     "raw": {
    #         "id": 37,
    #         "basecaller": "old",
    #     },
    #     "joint": {
    #         "id": 40,
    #         "basecaller": "old",
    #     },
    #     "event": {
    #         "id": 39,
    #         "basecaller": "old",
    #     },
    # },
    # (1, 1): {
    #     "raw": {
    #         "id": 36,
    #         "basecaller": "old",
    #     },
    #     "joint": {
    #         "id": 33,
    #         "basecaller": "old",
    #     },
    #     "event": {
    #         "id": 39,
    #         "basecaller": "old",
    #     },
    # },
}


def fix_all():
    me = MappingEvaluator()

    for k, v in test_results_meta.items():
        for d_type in v.keys():
            ENCODER_DEPTH = k[0]
            DECODER_DEPTH = k[1]
            if k == (1,1) or k == (2,1):
                test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.mask.{d_type}.encu128.encd{k[0]}.decu128.decd{k[1]}.{test_results_meta[k][d_type]['id']}.beam{BEAM_WIDTH}.json"
            else:
                test_results_meta[k][d_type]["path"] = f"info/snippets/mapping_evaluations/encd_{k[0]}_decd_{k[1]}/test/mapping_evaluator_results.snippets.test.{d_type}.{test_results_meta[k][d_type]['id']}.beam{BEAM_WIDTH}.json"

            print(f"Setting up {test_results_meta[k][d_type]['basecaller']} basecaller: encd_{k[0]}_decd_{k[1]}/model.1.{d_type}.lambda.mask.pad.lr0.0001.bilstm.encu128.encd{k[0]}.decu128.decd{k[1]}.b128.luong.tf0.5.strd6.spe10000.spv1500.{test_results_meta[k][d_type]['id']}")
            me.setup_basecaller(
                f'models/snippets/mask/encd_{k[0]}_decd_{k[1]}/model.1.{d_type}.lambda.mask.pad.lr0.0001.bilstm.encu128.encd{k[0]}.decu128.decd{k[1]}.b128.luong.tf0.5.strd6.spe10000.spv1500.{test_results_meta[k][d_type]["id"]}/model_chp',
                d_type,
                mode=test_results_meta[k][d_type]["basecaller"]
            )

            me.fix_results(Path(test_results_meta[k][d_type]["path"]))

if __name__ == '__main__':

    fix_all()
    # me = MappingEvaluator()

    # me.setup_basecaller('models/snippets/mask/encd_3_decd_2/model.1.joint.lambda.mask.pad.lr0.0001.bilstm.encu128.encd3.decu128.decd2.b128.luong.tf0.5.strd6.spe10000.spv1500.36/model_chp', 'joint')

    # with open('info/snippets/mapping_evaluations/encd_3_decd_2/test/mapping_evaluator_results.snippets.test.joint.36.ecoli.beam5.json', 'rt') as f:
    #     old_results = json.load(f)

    # fixed_results = []
    # for old_res in old_results:
    #     if old_res['read_length'] == 0:
    #         ident_read = me.run(old_res['path'])
    #         ident_read['path'] = old_res['path']
    #         ident_read['ref_length'] = np.loadtxt(old_res['path'].replace('.signal', '.label'), dtype=object).shape[0]
    #         print(ident_read)
    #         fixed_results.append(ident_read)

    #         with open('fixed.encd_3_decd_2.test.joint.36.ecoli.beam5.json', 'wt') as f:
    #             json.dump(fixed_results, f, indent=2)


