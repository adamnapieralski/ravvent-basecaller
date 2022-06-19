import numpy as np
import tensorflow as tf
from pathlib import Path

from timeit import default_timer as timer
import utils
import merger
import data_loader as dl
import merger
import basecaller as bc
import json

class RavventPerformanceEvaluator():
    def __init__(self, merger_scores_id=0):
        self.merger = merger.Merger(scores_id=merger_scores_id)
        self.stride = 6
        self.basecaller = None

    def _split_into_chunks(self, arr, def_chunk_size):
        """Splits array into chunks of def_chunk_size, the last chunk is left with the size <= def_chunk_size
        """
        return np.array_split(arr, np.arange(1, arr.shape[0] // def_chunk_size + 1) * def_chunk_size)

    def run(self, signal_data_source, chunk_size=1024):
        label_path = Path(signal_data_source).with_suffix('.label')
        labels = np.loadtxt(label_path, dtype='object')
        ranges_ids = labels[:,:2].astype(int)
        ref_seq = ''.join(list(labels[:,2]))

        samples_num = int(ranges_ids[-1,1] - ranges_ids[0,0])

        start = timer()
        raw_snippets, event_snippets, nuc_tk_snippets = dl.load_data_from_single_signal_label(signal_data_source, label_path, self.stride)

        raw_snippets_chunked = self._split_into_chunks(raw_snippets, chunk_size)
        event_snippets_chunked = self._split_into_chunks(event_snippets, chunk_size)
        nuc_tk_snippets_chunked = self._split_into_chunks(nuc_tk_snippets, chunk_size)


        data_chunks = []
        for raw_snippets, event_snippets, nuc_tk_snippets in zip(raw_snippets_chunked, event_snippets_chunked, nuc_tk_snippets_chunked):
            data_chunks.append((
                tf.convert_to_tensor(raw_snippets), tf.convert_to_tensor(event_snippets), tf.convert_to_tensor(nuc_tk_snippets)))

        t_data_loading = timer() - start
        nuc_preds = []

        t_predicting = 0
        t_postprocessing = 0

        for data in data_chunks:
            start = timer()
            input_data, target_data = utils.unpack_data_to_input_target(data, self.basecaller.input_data_type)

            pred_tokens, beam_scores = self.basecaller.beam_search_prediction(input_data, beam_width=5, max_output_len=tf.shape(target_data)[1])
            # pred_tokens, logits = self.basecaller.greedy_search_prediction(input_data, max_output_len=tf.shape(target_data)[1])

            # col_idx = tf.repeat(tf.expand_dims(tf.range(logits.shape[1]), axis=0), repeats=logits.shape[0], axis=0)
            # row_idx = tf.repeat(tf.expand_dims(tf.range(logits.shape[0]), axis=1), repeats=logits.shape[1], axis=-1)
            # idx = tf.stack([row_idx, col_idx, pred_tokens], axis=-1)
            # scores = tf.gather_nd(logits, idx).numpy()

            t_predicting += timer() - start
            start = timer()

            scores = utils.calc_prob_logits_beam_search_scores(beam_scores).numpy()
            seqs = self.basecaller.tokens_to_nuc_sequences(pred_tokens)

            nuc_preds.extend([
                merger.SeqLogitsPair(seq, list(sc[:len(seq)])) for seq, sc in zip(seqs, scores)])
            t_postprocessing += timer() - start

        start = timer()
        merged_seq = self.merger.merge(nuc_preds).seq
        t_merge = timer() - start


        return {
            'bases_num': len(ref_seq),
            'samples_num': samples_num,
            't_data_loading': t_data_loading,
            't_predicting': t_predicting,
            't_postprocessing': t_postprocessing,
            't_merge': t_merge,
            'total': t_data_loading + t_predicting + t_postprocessing + t_merge,
            'total_processing': t_predicting + t_postprocessing + t_merge
        }

    def setup_basecaller(self, weights_path, data_type, mode=1):
        if mode == 1:
            self.basecaller = bc.Basecaller(
                enc_units=128,
                dec_units=128,
                batch_sz=128,
                tokenizer=dl.nuc_tk,
                input_data_type=data_type,
                input_padding_value=dl.INPUT_PADDING,
                encoder_depth=2,
                decoder_depth=1,
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
        bases_num, samples_num = 0, 0
        t_total, t_processing = 0, 0
        with open(results_path, 'rt') as f:
            results = json.load(f)

        bases_speeds = []
        signals_speeds = []

        for res in results:
            bases_num += res['bases_num']
            samples_num += res['samples_num']
            t_total += res['total']
            t_processing += res['total_processing']

            bases_speeds.append(bases_num / t_processing)
            signals_speeds.append(samples_num / t_processing)


        return np.mean(bases_speeds), np.std(signals_speeds), np.mean(signals_speeds), np.std(signals_speeds)

        return bases_num / t_total, bases_num / t_processing, samples_num / t_total, samples_num / t_processing

    def evaluate_specific(self, files_info_path, results_path, weights_path, data_type):
        results = []
        self.setup_basecaller(weights_path, data_type, mode=1)

        with open(files_info_path, 'rt') as f:
            val_files = json.load(f)
            val_files = [v['signal_path'] for v in val_files]

        for v in val_files:
            print(f'Running {v}')
            res = self.run(v)
            res['path'] = v
            print(res)
            results.append(res)
            with open(results_path, 'wt') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    rpe = RavventPerformanceEvaluator()

    files_info_path = 'data/chiron/lambda/eval/all/files_info.val.snippets.stride_6.json'
    weights_paths = {
        'raw': 'models/snippets/model.1.raw.lambda.no_mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.40/model_chp',
        'joint': 'models/snippets/model.1.joint.lambda.no_mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.40/model_chp',
        'event': 'models/snippets/model.1.event.lambda.no_mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.36/model_chp'}


    # data_type = 'joint'
    # mode = 'cpu'
    # weights_path = weights_paths[data_type]

    # rpe.evaluate_specific(
    #     files_info_path, f'info/snippets/perform.{data_type}.{mode}.beam5.json', weights_path, data_type)

    # data_type = 'raw'
    # mode = 'cpu'
    # weights_path = weights_paths[data_type]

    # rpe.evaluate_specific(
    #     files_info_path, f'info/snippets/perform.{data_type}.{mode}.beam5.json', weights_path, data_type)

    # data_type = 'event'
    # mode = 'cpu'
    # weights_path = weights_paths[data_type]
    # rpe.evaluate_specific(
    #     files_info_path, f'info/snippets/perform.{data_type}.{mode}.beam5.json', weights_path, data_type)


    for d_type in ['event', 'raw', 'joint']:
        for comp in ['cpu', 'gpu']:
            res_1 = rpe.compute_total_results(f'info/snippets/performance/encd_3_decd_2/perform.{d_type}.{comp}1.encd3.decd2.beam1.json')
            # res_2 = rpe.compute_total_results(f'info/snippets/performance/encd_3_decd_2/perform.{d_type}.{comp}2.encd3.decd2.beam1.json')
            print(f'{d_type} - {comp}')
            print(res_1)
            print()
            # print(f'{d_type} - {comp}: {round(np.mean([res_1[1], res_2[1]]), 0)}+-{round(np.std([res_1[1], res_2[1]]), 0)}\t\t{round(np.mean([res_1[3], res_2[3]]), 0)}+-{round(np.std([res_1[3], res_2[3]]), 0)}')
            # print(f'{d_type} - {comp}: {round(np.mean([res_1[1], res_2[1]]), 0)}+-{round(np.std([res_1[1], res_2[1]]), 0)}\t\t{round(np.mean([res_1[3], res_2[3]]), 0)}+-{round(np.std([res_1[3], res_2[3]]), 0)}')

