import numpy as np
import matplotlib.pyplot as plt
import json, re, sys
from pathlib import Path

# import tensorflow as tf
# import analysis_utils as au
import pickle
import event_detection.event_detector as ed
import ravvent_mapping_evaluator as rme
# from data_loader import DataModule
# from basecaller import Basecaller
# import utils

def plot_raw_data():
    raw_id_min = 5000
    raw_id_max = 5400

    raw = np.loadtxt('data/chiron/ecoli/train/ecoli_0001.signal')
    labels = np.loadtxt('data/chiron/ecoli/train/ecoli_0001.label', dtype=object)
    range_ids = labels[:,:2].astype(int)
    nucs = labels[:, 2]
    fig, ax = plt.subplots(figsize=(15,4))
    ax.set_xlim([0, raw_id_max - raw_id_min])
    ax.set_xlabel('Signal ID')
    ax.set_ylabel('Current value [pA]')
    fig.tight_layout()

    raw = raw[raw_id_min:raw_id_max]
    ax.plot(raw)

    plt.savefig('figures/raw_data.png', dpi=300)

    id_start = np.argmax(range_ids[:,0] >= raw_id_min)
    id_end = np.argmax(np.logical_not(range_ids[:,0] < raw_id_max))

    split_ids = range_ids[id_start:id_end, 0]
    split_ids -= split_ids[0]

    nucs = nucs[id_start:id_end]

    for i, div in enumerate(split_ids):
        color = 'olive'
        ax.axvline(x=div, color=color)

        next_div = raw_id_max - raw_id_min if i == len(split_ids) - 1 else split_ids[i+1]

        symbol_x_pos = (div + next_div) / 2 - 1.5
        plt.text(symbol_x_pos, 220, nucs[i], transform=ax.transData)

    plt.savefig('figures/raw_data_marked.png', dpi=300)


def plot_reduced_raw_event_joint_test_accuracies_vs_no_of_6_mers():
    nums = [x / 4096 for x in [45, 450, 1024, 2048, 4096]]
    raw_accs = [0.9557888274973054, 0.9165415772299397, 0.9047021978693855, 0.8721022707489905, 0.7893045198856405]
    events_accs = [0.9499866626024884, 0.9103404033787701, 0.8924013682974483, 0.7982214934080496, 0.6285224738382291]
    joint_accs = [0.9648854692249131, 0.9315182947112179, 0.92731976799608, 0.9114789653329526, 0.7822268080455914]

    guppy_accs = [0.919906, 0.922886, 0.926774, 0.911608, 0.922477]
    fig, ax = plt.subplots(figsize=(6,4))

    lns_raw = ax.plot(nums[0:len(raw_accs)], raw_accs, label='raw', color='red')
    lns_event = ax.plot(nums[0:len(events_accs)], events_accs, label='event', color='blue')
    lns_joint = ax.plot(nums[0:len(joint_accs)], joint_accs, label='joint', color='green')
    lns_guppy = ax.plot(nums[0:len(guppy_accs)], guppy_accs, label='ONT guppy', color='purple', linestyle='dotted')

    ax.legend(loc='best')
    ax.set_xlabel('Fraction of all appearing 6-mers')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim((0.6, 1))
    ax.set_xlim((0, 1))
    # ax.set_xscale('log')
    # ax.set_title('Test accuracies of trained models on reduced datasets')
    ax.grid(axis='x')
    ax.grid(axis='y')
    # plt.show()
    plt.savefig('results/reduced_simulator_accuracies.png', dpi=300)

def plot_num_basic_6_mers_vs_all_appearing_6_mers():
    # with open('data/simulator-6-mers.json', 'rt') as f:
    #     mers = json.load(f)
    # sys.setrecursionlimit(5000)
    # _, nums = gsr.compute_number_of_appearing_mers(mers)
    with open('data/all-6-mers-nums-vs-basic.json', 'rt') as f:
        nums = json.load(f)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(nums)

    ax.set_xscale('log')

    xticks = [3, 12, 21, 43, 4096]
    yticks = [45, 450, 1024, 2048, 4096]

    for x_val in xticks:
        plt.axvline(x=x_val, c='grey', linewidth=0.5)
    for y_val in yticks:
        plt.axhline(y=y_val, c='grey', linewidth=0.5)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticks)

    # ax.set_title('Number of all appearing 6-mers vs number of basic 6-mers')
    ax.set_xlabel('Number of basic 6-mers')
    ax.set_ylabel('Number of all appearing 6-mers')
    # plt.show()
    plt.savefig('numbers_6_mers.png', dpi=300)


def plot_rnns_comparison():
    # gru, lstm, bigru, bilstm
    raw_accs = [0.8328752793495316, 0.8635220155922815, 0.9511955405603638, 0.9600662450181092]
    events_accs = [0.8095958504430516, 0.7980063058130534, 0.8652750877327555, 0.8480069291098759]
    joint_accs = [0.8145401713272199, 0.8726543102950517, 0.908816120906801, 0.9598024069354378]

    accs = [events_accs, raw_accs, joint_accs]
    gru_accs = [a[0] for a in accs]
    lstm_accs = [a[1] for a in accs]
    bigru_accs = [a[2] for a in accs]
    bilstm_accs = [a[3] for a in accs]

    labels = ['event', 'raw', 'joint']

    x = np.arange(len(labels))  # the label locations
    width = 0.18  # the width of the bars

    fig, ax = plt.subplots(figsize=(5,4))
    rects_gru = ax.bar(x - 1.5 * width, gru_accs, width, label='GRU')
    rects_lstm = ax.bar(x - width/2, lstm_accs, width, label='LSTM')
    rects_bigru = ax.bar(x + width/2, bigru_accs, width, label='BiGRU')
    rects_bilstm = ax.bar(x + 1.5 * width, bilstm_accs, width, label='BiLSTM')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy score')
    ax.set_xlabel('Input data type')

    # ax.set_title('Comparison of accuracies of RNN types')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')

    ax.bar_label(rects_gru, padding=3, label_type='center', rotation='vertical', fmt='%.3f')
    ax.bar_label(rects_lstm, padding=3, label_type='center', rotation='vertical', fmt='%.3f')
    ax.bar_label(rects_bigru, padding=3, label_type='center', rotation='vertical', fmt='%.3f')
    ax.bar_label(rects_bilstm, padding=3, label_type='center', rotation='vertical', fmt='%.3f')

    fig.tight_layout()

    # plt.show()
    plt.savefig('figures/rnns_comparison.png', dpi=300)

def plot_attention_weights():
    # # model_path = 'models/simulator.reduced/model.1.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.4096.600000.4096.81/model_chp'
    # # model_path = 'models/simulator.reduced/model.event.bilstm.u128.simulator.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.4096.600000.4096/model_chp'
    model_path = 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.4096.600000.4096/model_chp'

    data_path = 'data/simulator/reduced/seq.4096.600000.4096.new_padding.rawmax200.evmax30.b128.ed1.test.dataset'
    # data_path = 'data/simulator/reduced/seq.4096.600000.4096.rawmax200.evmax30.b128.ed1.test.dataset'
    ds = tf.data.experimental.load(data_path)

    for raw, ev, bases in ds.take(1):
        break

    # for i in range(50):
    #     print(i)
    au.plot_attention_weights_for_prediction(model_path=model_path, input_data=raw, output_max_length=30, seq_id=1, save_path=f'results/attention/raw/att.raw.png')

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,6))
    # img_event = plt.imread('results/attention/attention_weights_event.png')
    # img_raw = plt.imread('results/attention/attention_weights_raw.png')
    # img_joint = plt.imread('results/attention/attention_weights_joint.png')

    # fontsize = 8
    # ax1.set_axis_off()
    # ax1.imshow(img_event)
    # ax1.set_title('(a) event', fontsize=fontsize)

    # ax2.set_axis_off()
    # ax2.imshow(img_raw)
    # ax2.set_title('(b) raw', fontsize=fontsize)

    # ax3.set_axis_off()
    # ax3.imshow(img_joint)
    # ax3.set_title('(c) joint', fontsize=fontsize)

    # plt.tight_layout()
    # plt.savefig('results/attention/attention_weights_all_2.png', dpi=300, pad_inches=0)


def plot_event_detection_window_search():
    with open('data/chiron/lambda/train/ed_param_search.pkl', 'rb') as f:
        data_lambda = pickle.load(f)

    with open('data/chiron/ecoli/ed_param_search.pkl', 'rb') as f:
        data_ecoli = pickle.load(f)

    ref_len = 0
    detected = np.zeros((7, 9))

    for data in [data_lambda, data_ecoli]:
        for read_meta in data:
            ref_len += read_meta['ref_len']
            for k, v in read_meta['ed'].items():
                detected[k[0] - 3, (k[1] - 1) // 2 - 2] += v

    (detected - ref_len) / ref_len
    ref_error = (detected - ref_len) / ref_len
    ref_error_abs = abs(ref_error)

    title_fontsize = 16
    axis_labels_fontsize = 16
    ticks_fontsize = 14
    annot_fontsize = 14
    annot_fontsize_bold = 16


    ref_error_abs = abs(ref_error)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches((20, 7))
    im = ax1.imshow(ref_error_abs, cmap='viridis_r', vmin=0, vmax=1)
    cbar = ax1.figure.colorbar(im, ax=ax1)

    x_labels = [5,7,9,11,13,15,17,19,21]
    y_labels = [3,4,5,6,7,8,9]
    ax1.set_xticks(np.arange(len(x_labels)))
    ax1.set_yticks(np.arange(len(y_labels)))
    ax1.set_xticklabels(x_labels, fontsize=ticks_fontsize)
    ax1.set_yticklabels(y_labels, fontsize=ticks_fontsize)

    ax1.set_ylabel('$w_{len1}$', fontsize=axis_labels_fontsize)
    ax1.set_xlabel('$w_{len2}$', fontsize=axis_labels_fontsize)
    ax1.set_title('(a) whole analysed range', fontsize=title_fontsize)

    for i in range(ref_error_abs.shape[0]):
        for j in range(ref_error_abs.shape[1]):
            text = ax1.text(j, i, round(ref_error_abs[i, j], 3),
                        ha="center", va="center", color="w", fontsize=annot_fontsize)

    ref_error_cut = abs(ref_error[2:4, 1:6])
    im = ax2.imshow(ref_error_cut, cmap='viridis_r', vmin=0.085, vmax=0.1)

    cbar = ax2.figure.colorbar(im, ax=ax2)

    x_labels = [7,9,11,13,15]
    y_labels = [5, 6]
    ax2.set_xticks(np.arange(len(x_labels)))
    ax2.set_yticks(np.arange(len(y_labels)))
    ax2.set_xticklabels(x_labels, fontsize=ticks_fontsize)
    ax2.set_yticklabels(y_labels, fontsize=ticks_fontsize)

    ax2.set_ylabel('$w_{len1}$', fontsize=axis_labels_fontsize)
    ax2.set_xlabel('$w_{len2}$', fontsize=axis_labels_fontsize)
    ax2.set_title('(b) best fitting range', fontsize=title_fontsize)


    for i in range(ref_error_cut.shape[0]):
        for j in range(ref_error_cut.shape[1]):
            if i == 1 and j == 1:
                text = ax2.text(j, i, round(ref_error_cut[i, j], 5), ha="center", va="center", color="w", weight='bold', fontsize=annot_fontsize_bold)
            else:
                text = ax2.text(j, i, round(ref_error_cut[i, j], 5), ha="center", va="center", color="w", fontsize=annot_fontsize)

    plt.tight_layout()

    plt.savefig('event_detection_window_search.png', dpi=600)


def plot_event_detection():
    short_win_len = 6
    long_win_len = 16
    detector = ed.EventDetector(window_length1=short_win_len, window_length2=long_win_len)
    detector.reset()

    raw = np.loadtxt('data/chiron/ecoli_0001_0080/ecoli_0001.signal')
    raw = raw[20200:20500]

    tstat1 = detector.compute_tstat_all(raw, detector.short_detector['window_length'])
    tstat2 = detector.compute_tstat_all(raw, detector.long_detector['window_length'])

    detector.short_detector['tstat'] = tstat1
    detector.long_detector['tstat'] = tstat2

    peaks = detector.detect_peak_all()

    raw = raw[long_win_len+1:-long_win_len-1]
    tstat1 = tstat1[long_win_len+1:-long_win_len-1]
    tstat2 = tstat2[long_win_len+1:-long_win_len-1]
    peaks -= long_win_len + 1
    peaks = peaks[peaks > 0]
    peaks = peaks[peaks < len(raw)]

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches((15, 10))
    ax1.plot(raw)
    ax1.set_title('(a) Raw data')
    ax1.set_xlabel('Raw data id')
    ax1.set_ylabel('Value')


    ax2.plot(tstat1)
    ax2.set_title('(b) T-test$_1$ data')
    ax2.set_xlabel('Raw data id')
    ax2.set_ylabel('Value')


    ax3.plot(tstat2)
    ax3.set_title('(b) T-test$_2$ data')
    ax3.set_xlabel('Raw data id')
    ax3.set_ylabel('Value')


    for div in peaks:
        color = 'olive'
        ax1.axvline(x=div, color=color)
        ax2.axvline(x=div, color=color)
        ax3.axvline(x=div, color=color)

    fig.tight_layout()
    plt.savefig('event_detection.png', dpi=600)


def plot_learning_curves():
    me = rme.MappingEvaluator()

    results_path = Path('info/snippets/mapping_evaluations')

    event_res_map = [p for p in results_path.iterdir() if re.match(r'.*event\.\d\d\.json$', str(p))]
    event_res_map.sort()
    event_res_map = np.array([me.compute_total_results(str(p)) for p in event_res_map])

    joint_res_map = [p for p in results_path.iterdir() if re.match(r'.*joint\.\d\d\.json$', str(p))]
    joint_res_map.sort()
    joint_res_map = np.array([me.compute_total_results(str(p)) for p in joint_res_map])

    raw_res_map = [p for p in results_path.iterdir() if re.match(r'.*raw\.\d\d\.json$', str(p))]
    raw_res_map.sort()
    raw_res_map = np.array([me.compute_total_results(str(p)) for p in raw_res_map])

    event_log_path = Path('info/snippets/csvlog.1.event.lambda.mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.log')
    joint_log_path = Path('info/snippets/csvlog.1.joint.lambda.mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.log')
    raw_log_path = Path('info/snippets/csvlog.1.raw.lambda.mask.pad.lr0.0001.bilstm.encu128.encd2.decu128.decd1.b128.luong.tf0.5.strd6.spe10000.spv1500.log')

    event_logs = np.loadtxt(event_log_path, dtype=object, delimiter=',')[1:,1:].astype('float')
    joint_logs = np.loadtxt(joint_log_path, dtype=object, delimiter=',')[1:,1:].astype('float')
    raw_logs = np.loadtxt(raw_log_path, dtype=object, delimiter=',')[1:,1:].astype('float')

    data = {
        'joint': {
            'log_acc': joint_logs[:, 0],
            'log_val_acc': joint_logs[:, 2],
            'log_loss': joint_logs[:, 1],
            'log_val_loss': joint_logs[:, 3],
            'map_acc': joint_res_map[:,0] / 100
        },
        'raw': {
            'log_acc': raw_logs[:, 0],
            'log_val_acc': raw_logs[:, 2],
            'log_loss': raw_logs[:, 1],
            'log_val_loss': raw_logs[:, 3],
            'map_acc': raw_res_map[:,0] / 100
        },
        'event': {
            'log_acc': event_logs[:, 0],
            'log_val_acc': event_logs[:, 2],
            'log_loss': event_logs[:, 1],
            'log_val_loss': event_logs[:, 3],
            'map_acc': event_res_map[:,0] / 100
        }
    }

    for data_type in ['joint', 'raw', 'event']:
        fig, ax_loss = plt.subplots(figsize=(6.4, 4))
        ax_acc = ax_loss.twinx()

        log_loss, = ax_loss.plot(data[data_type]['log_loss'], 'r', linewidth=1, label='train loss')
        # log_val_loss, = ax_loss.plot(data[data_type]['log_val_loss'], 'b--', linewidth=1, label='val. loss')

        # log_acc, = ax_acc.plot(data[data_type]['log_acc'], 'r', linewidth=1, label='train acc.')
        log_val_acc, = ax_acc.plot(data[data_type]['log_val_acc'], 'b', linewidth=1, label='val. subset acc.')

        map_acc, = ax_acc.plot(data[data_type]['map_acc'], color='g', linewidth=1.5, label='val. read acc.')

        ax_acc.set_ylim((0, 1))
        # ax_acc.set_xlim((0, 40))
        ax_loss.set_xlabel('Epoch number')
        ax_acc.set_ylabel('Accuracy')
        ax_loss.set_ylabel('Loss')

        ax_acc.legend(handles=[log_loss, log_val_acc, map_acc], loc='lower right', framealpha=0.5)

        fig.tight_layout()
        plt.savefig(f'figures/learning_curve_{data_type}.png', dpi=600)


if __name__ == '__main__':
    # plot_raw_data()
    # plot_reduced_raw_event_joint_test_accuracies_vs_no_of_6_mers()
    # plot_num_basic_6_mers_vs_all_appearing_6_mers()
    # plot_rnns_comparison()
    # plot_attention_weights()
    # plot_event_detection_window_search()
    # plot_event_detection()
    plot_learning_curves()