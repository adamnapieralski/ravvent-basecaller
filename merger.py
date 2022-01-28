from Bio import pairwise2
from logzero import logger

from typing import Any, Dict, List, Type


class SeqLogitsPair(object):

    @classmethod
    def align_logits(
            cls,
            seq_gapped: str,
            logits_non_gapped: List[float],
    ) -> List[float]:
        logits_gapped = []  # type: List[float]
        index = 0
        for c in seq_gapped:
            if c == '-':
                logits_gapped.append(-1.)
            else:
                logits_gapped.append(logits_non_gapped[index])
                index += 1
        return logits_gapped

    @property
    def seq(self) -> str:
        return self._seq

    @property
    def logits(self) -> List[float]:
        return self._logits

    def __init__(self, seq: str, logits: List[float]) -> None:
        assert(len(seq) == len(logits))
        self._seq = seq
        self._logits = logits
        return

class MergerLeftPriority():

    def __init__(self) -> None:
        return

    def __get_end_index(
            self,
            seq: str,
    ) -> int:
        for index in range(len(seq) - 1, -1, -1):
            if seq[index] != '-':
                return index
        raise ValueError

    def merge(
            self,
            seq_logits_pair1: SeqLogitsPair,
            seq_logits_pair2: SeqLogitsPair,
    ) -> SeqLogitsPair:
        seq1 = seq_logits_pair1.seq
        seq2 = seq_logits_pair2.seq
        logits1 = seq_logits_pair1.logits
        logits2 = seq_logits_pair2.logits
        assert(len(seq1) == len(seq2))
        end_index_seq1 = self.__get_end_index(seq1)
        seq_merged_gapped = (
            seq1[:end_index_seq1 + 1] +
            seq2[end_index_seq1 + 1:]
        )
        logits_merged_gapped = (
            logits1[:end_index_seq1 + 1] +
            logits2[end_index_seq1 + 1:]
        )
        assert(len(seq_merged_gapped) == len(logits_merged_gapped))
        seq_merged = seq_merged_gapped.replace('-', '')
        logits_merged = [
            score for score in logits_merged_gapped
            if score > 0
        ]
        return SeqLogitsPair(
            seq=seq_merged,
            logits=logits_merged,
        )

class SingleMergerByLogits():

    def __init__(self) -> None:
        return

    def merge(
            self,
            seq_logits_pair1: SeqLogitsPair,
            seq_logits_pair2: SeqLogitsPair,
    ) -> SeqLogitsPair:
        seq1 = seq_logits_pair1.seq
        seq2 = seq_logits_pair2.seq
        logits1 = seq_logits_pair1.logits
        logits2 = seq_logits_pair2.logits
        assert(len(seq1) == len(seq2))

        seq_merged = ''
        logits_merged = []
        for n1, n2, l1, l2 in zip(seq1, seq2, logits1, logits2):
            if n1 == '-':
                seq_merged += n2
                logits_merged.append(l2)
            elif n2 == '-':
                seq_merged += n1
                logits_merged.append(l1)
            elif l2 > l1:
                seq_merged += n2
                logits_merged.append(l2)
            else:
                seq_merged += n1
                logits_merged.append(l1)

        assert(len(seq_merged) == len(seq1))
        return SeqLogitsPair(
            seq=seq_merged,
            logits=logits_merged,
        )

class Merger():

    def __init__(self, scores_id=0) -> None:
        self.scores = {
            0: {
                'match_score': 1.,
                'mismatch_score': -1.,
                'gap_open_score': -1.,
                'gap_extend_score': -0.2
            },
            1: {
                'match_score': 5.,
                'mismatch_score': -4.,
                'gap_open_score': -3.,
                'gap_extend_score': -.1
            },
            2: {
                'matrix': {
                    ('A','A'): 10., ('A','C'): -3., ('A','G'): -1., ('A','T'): -4.,
                    ('C','A'): -3., ('C','C'): 9., ('C','G'): -5., ('C','T'): 0.,
                    ('G','A'): -1., ('G','C'): -5., ('G','G'): 7., ('G','T'): -3.,
                    ('T','A'): -4., ('T','C'): 0., ('T','G'): -3., ('T','T'): 8.
                },
                'gap_open_score': -9.,
                'gap_extend_score': -2.
            }
        }
        self.scores_id = scores_id

        self.overlap_seq_len = 25

        self._merger = SingleMergerByLogits()


    def merge(self, nuc_pred_snippets):
        seq_merged = nuc_pred_snippets[0].seq
        logits_merged = nuc_pred_snippets[0].logits
        merge_flag = False

        for i in range(1, len(nuc_pred_snippets)):
            seq_appended = nuc_pred_snippets[i].seq
            logits_appended = nuc_pred_snippets[i].logits
            seq1_overlap = seq_merged[-self.overlap_seq_len:]
            seq2_overlap = seq_appended[:self.overlap_seq_len]
            logits1_overlap = logits_merged[-self.overlap_seq_len:]
            logits2_overlap = logits_appended[:self.overlap_seq_len]
            if self.scores_id in [0, 1]:
                algns = pairwise2.align.localms(
                    seq1_overlap,
                    seq2_overlap,
                    *list(self.scores[self.scores_id].values())
                )
            elif self.scores_id == 2:
                algns = pairwise2.align.localds(
                    seq1_overlap,
                    seq2_overlap,
                    self.scores[self.scores_id]['matrix'],
                    self.scores[self.scores_id]['gap_open_score'],
                    self.scores[self.scores_id]['gap_extend_score'],
                )
            if len(algns) == 0:
                logger.warning(
                    'no alignment was found between {}th and {}th snippets'.format(
                        i - 1,
                        i,
                    ))
                if not merge_flag:
                    seq_merged = nuc_pred_snippets[i].seq
                    logits_merged = nuc_pred_snippets[i].logits
                    logger.debug('alignment does not still exist')
                    continue
                else:
                    logger.debug('merged seq already found, so that is returned')
                    return SeqLogitsPair(
                        seq=seq_merged,
                        logits=logits_merged,
                    )
            else:
                merge_flag = True
                a = algns[0]
                seq1_gapped = a[0]
                seq2_gapped = a[1]
                alignment_score = a[2]
                merge_len = (
                    len(seq1_gapped) -
                    seq1_gapped.count('-') -
                    seq2_gapped.count('-')
                )

                # logger.debug('align 1: {}'.format(seq1_gapped))
                # logger.debug('align 2: {}'.format(seq2_gapped))

                logits1_gapped = SeqLogitsPair.align_logits(
                    seq_gapped=seq1_gapped,
                    logits_non_gapped=logits1_overlap,
                )
                logits2_gapped = SeqLogitsPair.align_logits(
                    seq_gapped=seq2_gapped,
                    logits_non_gapped=logits2_overlap,
                )

                seq_logits_pair1 = SeqLogitsPair(
                    seq=seq1_gapped,
                    logits=logits1_gapped,
                )
                seq_logits_pair2 = SeqLogitsPair(
                    seq=seq2_gapped,
                    logits=logits2_gapped,
                )
                seq_logits_pair_merged = self._merger.merge(
                    seq_logits_pair1,
                    seq_logits_pair2,
                )

                seq_merged = (
                    seq_merged[:-self.overlap_seq_len] +
                    seq_logits_pair_merged.seq +
                    seq_appended[self.overlap_seq_len:]
                )
                logits_merged = (
                    logits_merged[:-self.overlap_seq_len] +
                    seq_logits_pair_merged.logits +
                    logits_appended[self.overlap_seq_len:]
                )
        return SeqLogitsPair(
            seq=seq_merged,
            logits=logits_merged,
        )


if __name__ == '__main__':
    m = Merger()
    seq1, seq2 = 'AGTTCAGCGATCGGATCCGCGTGC', 'GAGATTTTATCCGCGTGCTGTTTACG'
    n1 = SeqLogitsPair(seq1, [0.5] * len(seq1))
    n2 = SeqLogitsPair(seq2, [0.7] * len(seq2))
    out = m.merge([n1, n2])
    print(out.seq, out.logits)