import numpy as np
import matplotlib.pyplot as plt
import json, re, sys
from pathlib import Path

import tensorflow as tf
import analysis_utils as au
# from data_loader import DataModule
# from basecaller import Basecaller
# import utils

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
    raw_accs = [0.6167816019985375, 0.6338807466942893, 0.6384401330306355, 0.6722500736512388]
    events_accs = [0.5328741160783943, 0.5431880844269427, 0.5785425703314027, 0.5797833170351909]
    joint_accs = [0.5452239584748999, 0.5334056789666597, 0.6071774475674491, 0.6288333857969358]

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
    ax.set_ylabel('Accuracy scores')
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
    plt.savefig('rnns_comparison.png', dpi=300)

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
if __name__ == '__main__':
    plot_reduced_raw_event_joint_test_accuracies_vs_no_of_6_mers()
    # plot_num_basic_6_mers_vs_all_appearing_6_mers()
    # plot_rnns_comparison()
    # plot_attention_weights()