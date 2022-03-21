"""Classes for performing event detection on nanopore raw signal data
Based on:
- Scrappie (past ONT basecaller) https://github.com/nanoporetech/scrappie
- event_detection https://github.com/skovaka/event_detection
"""

import numpy as np
import math

FLT_MIN = 1.17549435e-38
FLT_MAX = 3.40282347e+38


class Event():
    def __init__(self, start: int, length: int, mean: float, stdv: float) -> None:
        self.start = start
        self.length = length
        self.mean = mean
        self.stdv = stdv

    @property
    def end(self) -> int:
        return self.start + self.length


class EventDetector():
    def __init__(self, window_length1=3, window_length2=6, threshold1=1.4,
                 threshold2=9., peak_height=0.2):
        self.params = {
            'window_length1': window_length1, 'window_length2': window_length2,
            'threshold1': threshold1, 'threshold2': threshold2, 'peak_height': peak_height,
            # 'min_mean': min_mean, 'max_mean': max_mean
        }
        self.BUF_LEN = 1 + self.params['window_length2'] * 2
        self.sum = np.zeros(shape=(self.BUF_LEN), dtype=np.float64)
        self.sumsq = np.zeros(shape=(self.BUF_LEN), dtype=np.float64)

        self._event = Event(0, 0, 0, 0)

        self.reset()

    def reset(self):
        self.sum[0], self.sumsq[0] = 0., 0.
        self.t, self.evt_st, self.evt_st_sum, self.evt_st_sumsq = 1, 0, 0., 0.

        self.short_detector = {
            'DEF_PEAK_POS': -1,
            'DEF_PEAK_VAL': FLT_MAX,
            'threshold': self.params['threshold1'],
            'window_length': self.params['window_length1'],
            'masked_to': 0,
            'peak_pos': -1,
            'peak_value': FLT_MAX,
            'valid_peak': False
        }

        self.long_detector = {
            'DEF_PEAK_POS': -1,
            'DEF_PEAK_VAL': FLT_MAX,
            'threshold': self.params['threshold2'],
            'window_length': self.params['window_length2'],
            'masked_to': 0,
            'peak_pos': -1,
            'peak_value': FLT_MAX,
            'valid_peak': False
        }

    @property
    def event(self):
        return self._event

    def get_buf_mid(self):
        return to_u32(self.t - int(self.BUF_LEN / 2) - 1)

    def run(self, raw):
        events = []

        for i in range(raw.size):
            if (self._add_sample(raw[i])):
                events.append(self._event)

        self.reset()
        return events

    def _add_sample(self, s):
        t_mod = to_u32(self.t % self.BUF_LEN)
        if t_mod > 0:
            self.sum[t_mod] = self.sum[t_mod-1] + s
            self.sumsq[t_mod] = self.sumsq[t_mod-1] + s*s
        else:
            self.sum[t_mod] = self.sum[self.BUF_LEN-1] + s
            self.sumsq[t_mod] = self.sumsq[self.BUF_LEN-1] + s*s

        self.t = to_u32(self.t + 1)
        self.buf_mid = self.get_buf_mid()
        tstat1 = self._compute_tstat(self.params['window_length1'])
        tstat2 = self._compute_tstat(self.params['window_length2'])

        p1 = self._detect_peak(tstat1, self.short_detector)
        p2 = self._detect_peak(tstat2, self.long_detector)

        if p1 or p2:
            return self._create_event(
                self.buf_mid - self.params['window_length1'] + 1)
            # return self._event.mean >= self.params['min_mean'] and self._event.mean <= self.params['max_mean']

        return False

    def _compute_tstat(self, w_length):
        assert w_length > 0
        eta = FLT_MIN
        w_lengthf = float(w_length)

        # Quick return:
        # t-test not defined for number of points less than 2
        # need at least as many points as twice the window length
        if self.t <= 2 * w_length or w_length < 2:
            return 0

        # fughe boundaries
        # for i in range(w_length):
        #     self.tstat[i] = 0
        #     self.tstat[self.d_length - i - 1] = 0

        i = to_u32(self.buf_mid % self.BUF_LEN)
        st = to_u32(self.buf_mid - w_length) % self.BUF_LEN
        en = to_u32(self.buf_mid + w_length) % self.BUF_LEN

        # print('{}\t{}\t{}'.format(i, st, en))

        sum1 = self.sum[i] - self.sum[st]
        sumsq1 = self.sumsq[i] - self.sumsq[st]
        sum2 = self.sum[en] - self.sum[i]
        sumsq2 = self.sumsq[en] - self.sumsq[i]
        mean1, mean2 = sum1 / w_lengthf, sum2 / w_lengthf
        combined_var = sumsq1 / w_lengthf - mean1 * \
            mean1 + sumsq2 / w_lengthf - mean2 * mean2

        # Prevent problem due to very small variances
        combined_var = max(combined_var, eta)

        # t-stat
        # Formula is a simplified version of Student's t-statistic for the
        # special case where there are two samples of equal size with
        # differing variance
        delta_mean = mean2 - mean1
        return math.fabs(delta_mean) / math.sqrt(combined_var / w_lengthf)

    def _detect_peak(self, current_value, detector):
        if detector['masked_to'] >= self.buf_mid:
            return False

        if detector['peak_pos'] == detector['DEF_PEAK_POS']:
            # Case 1: we've not yet recorder maximum
            if current_value < detector['peak_value']:
                # either record a deeper minimum
                detector['peak_value'] = current_value
            elif current_value - detector['peak_value'] > self.params['peak_height']:
                # or we've seen a qualifying maximum
                detector['peak_value'] = current_value
                detector['peak_pos'] = to_i32(self.buf_mid)
                # otherwise wait to rise high enough to be considered a peak
        else:
            # Case 2: In an existing peak, waiting to see if it's good
            if current_value > detector['peak_value']:
                # Update the peak
                detector['peak_value'] = current_value
                detector['peak_pos'] = to_i32(self.buf_mid)
            # Dominate other tstat signals if we're going to fire at some point
            if detector['window_length'] == self.short_detector['window_length']:
                if detector['peak_value'] > detector['threshold']:
                    self.long_detector['masked_to'] = to_u32(
                        detector['peak_pos'] + detector['window_length'])
                    self.long_detector['peak_pos'] = self.long_detector['DEF_PEAK_POS']
                    self.long_detector['peak_value'] = self.long_detector['DEF_PEAK_VAL']
                    self.long_detector['valid_peak'] = False
            # if we convinced ourselves we've seen a peak
            if detector['peak_value'] - current_value > self.params['peak_height'] and detector['peak_value'] > detector['threshold']:
                detector['valid_peak'] = True
            # Finally, check the distance if this is a good peak
            if detector['valid_peak'] and (self.buf_mid - detector['peak_pos']) > detector['window_length'] / 2:
                detector['peak_pos'] = detector['DEF_PEAK_POS']
                detector['peak_value'] = current_value
                detector['valid_peak'] = False

                return True
        return False

    def _create_event(self, evt_en):
        evt_en = to_u32(evt_en)
        evt_en_buf = to_u32(evt_en % self.BUF_LEN)
        start_ = self.evt_st
        length_ = float(evt_en - self.evt_st)
        if length_ < FLT_MIN:
            return False
        mean_ = float(
            self.sum[evt_en_buf] - self.evt_st_sum) / length_

        deltasqr = self.sumsq[evt_en_buf] - self.evt_st_sumsq
        stdv_ = deltasqr / length_ - mean_ ** 2
        stdv_ = math.sqrt(max(stdv_, FLT_MIN))

        length_ = int(length_)

        self._event = Event(start_, length_, mean_, stdv_)

        self.evt_st = evt_en
        self.evt_st_sum = self.sum[evt_en_buf]
        self.evt_st_sumsq = self.sumsq[evt_en_buf]
        return True

    def compute_tstat_all(self, raw, w_len):
        sum = np.cumsum(raw)
        sumsq = np.cumsum(raw*raw)

        tstat = np.zeros_like(sum)

        for i in range(w_len, len(sum) - w_len):
            sum1 = sum[i]
            sumsq1 = sumsq[i]
            if i > w_len:
                sum1 -= sum[i-w_len]
                sumsq1 -= sumsq[i-w_len]
            sum2 = sum[i+w_len] - sum[i]
            sumsq2 = sumsq[i+w_len] - sumsq[i]
            mean1 = sum1 / w_len
            mean2 = sum2 / w_len
            combined_var = sumsq1 / w_len - mean1 * mean1 + sumsq2 / w_len - mean2 * mean2
            combined_var = max((combined_var, FLT_MIN))
            delta_mean = mean2 - mean1
            tstat[i] = abs(delta_mean) / ((combined_var / w_len) ** 0.5)
        return tstat

    def detect_peak_all(self):
        detectors = [self.short_detector, self.long_detector]

        peaks = np.zeros_like(self.short_detector['tstat'])
        peak_count = 0
        for i in range(len(peaks)):
            for detector in detectors:
                if detector['masked_to'] >= i:
                    continue

                current_value = detector['tstat'][i]

                if detector['peak_pos'] == detector['DEF_PEAK_POS']:
                    # Case 1: we've not yet recorder maximum
                    if current_value < detector['peak_value']:
                        # either record a deeper minimum
                        detector['peak_value'] = current_value
                    elif current_value - detector['peak_value'] > self.params['peak_height']:
                        # or we've seen a qualifying maximum
                        detector['peak_value'] = current_value
                        detector['peak_pos'] = i
                        # otherwise wait to rise high enough to be considered a peak
                else:
                    # Case 2: In an existing peak, waiting to see if it's good
                    if current_value > detector['peak_value']:
                        # Update the peak
                        detector['peak_value'] = current_value
                        detector['peak_pos'] = i
                    # Dominate other tstat signals if we're going to fire at some point
                    if detector == self.short_detector:
                        if detector['peak_value'] > detector['threshold']:
                            self.long_detector['masked_to'] = detector['peak_pos'] + detector['window_length']
                            self.long_detector['peak_pos'] = self.long_detector['DEF_PEAK_POS']
                            self.long_detector['peak_value'] = self.long_detector['DEF_PEAK_VAL']
                            self.long_detector['valid_peak'] = False
                    # if we convinced ourselves we've seen a peak
                    if detector['peak_value'] - current_value > self.params['peak_height'] and detector['peak_value'] > detector['threshold']:
                        detector['valid_peak'] = True
                    # Finally, check the distance if this is a good peak
                    if detector['valid_peak'] and (i - detector['peak_pos']) > detector['window_length'] / 2:
                        peaks[peak_count] = detector['peak_pos']
                        peak_count += 1
                        detector['peak_pos'] = detector['DEF_PEAK_POS']
                        detector['peak_value'] = current_value
                        detector['valid_peak'] = False
        return peaks[peaks > 0].astype(int)

def to_u32(val):
    return int(val) & 0xffffffff


def to_i32(val):
    n = val & 0xffffffff
    return (n ^ 0x80000000) - 0x80000000

