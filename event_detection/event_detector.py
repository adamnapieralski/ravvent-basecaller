import numpy as np
import math

FLT_MIN = 1e-9
FLT_MAX = 3.402823e+38

class EventDetector():
    def __init__(self, window_length1=3, window_length2=6, threshold1=1.4, threshold2=9., peak_height=0.2,
                min_mean=30, max_mean=150):
        self.params = {
            'window_length1': window_length1, 'window_length2': window_length2,
            'threshold1': threshold1, 'threshold2': threshold2,
            'peak_height': peak_height, 'min_mean': min_mean, 'max_mean': max_mean
        }
        self.BUF_LEN = 1 + self.params['window_length2'] * 2
        self.sum = np.zeros(shape=(self.BUF_LEN))
        self.sumsq = np.zeros(shape=(self.BUF_LEN))

        self.event_ = { 'length': 0, 'mean': 0, 'start': 0, 'stdv': 0 }

        self.reset()

    def reset(self):
        self.sum[0], self.sumsq[0] = 0., 0.
        self.t, self.evt_st, self.evt_st_sum, self.evt_st_sumsq = 1, 0, 0., 0.

        self.short_detector = {
            'DEF_PEAK_POS': -1,
            'DEF_PEAK_VAL': FLT_MAX,
            'threshold': self.params['window_length1'],
            'masked_to': 0,
            'peak_pos': -1,
            'peak_value': FLT_MAX,
            'valid_peak': False
        }

        self.long_detector = {
            'DEF_PEAK_POS': -1,
            'DEF_PEAK_VAL': FLT_MAX,
            'threshold': self.params['window_length2'],
            'masked_to': 0,
            'peak_pos': -1,
            'peak_value': FLT_MAX,
            'valid_peak': False
        }

    def get_buf_mid(self):
        return self.t - (self.BUF_LEN / 2) - 1

    def get(self):
        return self.event_

    def add_samples(self, raw):
        events = [] #np.zeros(shape=(raw.size / self.params['window_length2']))

        for i in range(raw.size):
            if (self.add_sample(raw[i])):
                events.append(self.event_)

        self.reset()
        return events

    def add_sample(self, s):
        t_mod = self.t % self.BUF_LEN
        if t_mod > 0:
            self.sum[t_mod] = self.sum[t_mod-1] + s
            self.sumsq[t_mod] = self.sumsq[t_mod-1] + s*s
        else:
            self.sum[t_mod] = self.sum[self.BUF_LEN-1] + s
            self.sumsq[t_mod] = self.sumsq[self.BUF_LEN-1] + s*s

        self.t += 1
        self.buf_mid = self.get_buf_mid()
        tstat1 = self.compute_tstat(self.params['window_length1'])
        tstat2 = self.compute_tstat(self.params['window_length2'])

        p1 = self.peak_detect(tstat1, self.short_detector)
        p2 = self.peak_detect(tstat2, self.long_detector)

        if p1 or p2:
            self.create_event(self.buf_mid - self.params['window_length1'] + 1)
            return self.event_['mean'] >= self.params['min_mean'] and self.event_['mean'] <= self.params['max_mean']

        return False

    def compute_tstat(self, w_length):
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

        i = int(self.buf_mid % self.BUF_LEN)
        st = (self.buf_mid - w_length) % self.BUF_LEN
        en = (self.buf_mid + w_length) % self.BUF_LEN

        print(i, type(i))
        sum1 = self.sum[i] - self.sum[st]
        sumsq1 = self.sumsq[i] - self.sumsq[st]
        sum2 = float(self.sum[en] - self.sum[i])
        sumsq2 = float(self.sumsq[en] - self.sumsq[i])
        mean1, mean2 = sum1 / w_lengthf, sum2 / w_lengthf
        combined_var = sumsq1 / w_lengthf - mean1 * mean1 + sumsq2 / w_lengthf - mean2 * mean2

        # Prevent problem due to very small variances
        combined_var = max(combined_var, eta)

        # t-stat
        # Formula is a simplified version of Student's t-statistic for the
        # special case where there are two samples of equal size with
        # differing variance
        delta_mean = mean2 - mean1
        return math.fabs(delta_mean) / math.sqrt(combined_var / w_lengthf)

    def peak_detect(self, current_value, detector):
        i = self.buf_mid

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
                detector['peak_pos'] = self.buf_mid
                # otherwise wait to rise high enough to be considered a peak
        else:
            # Case 2: In an existing peak, waiting to see if it's good
            if current_value > detector['peak_value']:
                # Update the peak
                detector['peak_value'] = current_value
                detector['peak_pos'] = self.buf_mid
            # Dominate other tstat signals if we're going to fire at some point
            if detector['window_length'] == self.short_detector['window_length']:
                if detector['peak_value'] > detector['threshold']:
                    self.long_detector['masked_to'] = detector['peak_pos'] + detector['window_length']
                    self.long_detector['peak_pos'] = self.long_detector['DEF_PEAK_POS']
                    self.long_detector['peak_value'] = self.long_detector['DEF_PEAK_VAL']
                    self.long_detector['valid_peak'] = False
            # if we convinced ourselves we've seen a peak
            if detector['peak_value'] - current_value > self.params['peak_height'] and detector['peak_value'] > detector['threshold']:
                detector['valid_peak'] = True
            # Finally, check the distance if this is a good peak
            if detector['valid_peak'] and (self.buf_mid - detector['peak_pos'] > detector['window_length'] / 2):
                detector['peak_pos'] = detector['DEF_PEAK_POS']
                detector['peak_value'] = current_value
                detector['valid_peak'] = False

                return True
        return False

    def create_event(self, evt_en):
        evt_en_buf = evt_en % self.BUF_LEN
        self.event_['start'] = self.evt_st
        self.event_['length'] = float(evt_en - self.evt_st)
        self.event_['mean'] = float(self.sum[evt_en_buf] - self.evt_st_sum) / self.event_['length']

        deltasqr = self.sumsq[evt_en_buf] - self.evt_st_sumsq
        var = deltasqr / self.event_['length'] - self.event_['mean'] ** 2
        self.event_['stdv'] = math.sqrt(max(var, FLT_MAX))

        self.evt_st = evt_en
        self.evt_st_sum = self.sum[evt_en_buf]
        self.evt_st_sumsq = self.sumsq[evt_en_buf]

if __name__ == '__main__':
    ed = EventDetector(window_length2=13)

    raw = np.loadtxt('data/chiron/ecoli_0001_0080/ecoli_0001.signal')
    print(type(raw))
    raw_start, raw_len = 0, len(raw)

    for i in range(raw_start, raw_start + raw_len):
        if (ed.add_sample(raw[i])):
            e = ed.get()
            print('{}\t{}\t{}\t{}'.format(e['mean'], e['stdv'], e['length'], raw_start+e['start']))