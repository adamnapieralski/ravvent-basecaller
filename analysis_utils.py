"""
Various utils functions for output analysis
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json, re
from pathlib import Path
from typing import Tuple
from pathlib import Path

from data_loader import DataModule
from basecaller import Basecaller
import utils

def create_train_history_figure(info_path: str, save_path: str = None, loss_lim: Tuple[float, float] = None, accuracy_lim: Tuple[float, float] = (0, 1), figsize: Tuple[int,int] = (6,4)):
    """Create (and display/save) figure with loss, val_loss and val_accuracy from training.

    Args:
        info_path (str): Path to info file with training history
        save_path (str, optional): Where to save figure - if None, it is displayed. Defaults to None.
    """

    with open(info_path, 'r') as f:
        info = json.load(f)
    history = info['train_history']

    # extract title from info path file
    res = re.match(r'.*info\.(.*)\.json', info_path)
    title = res.group(1)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    lns1 = ax1.plot(history['batch_loss'], label='loss', color='red')
    lns2 = ax1.plot(history['val_batch_loss'], label='val_loss', color='blue')

    min_val_loss_epoch = history['val_batch_loss'].index(min(history['val_batch_loss']))
    plt.axvline(x=min_val_loss_epoch, c='grey')

    ax2 = ax1.twinx()

    ax2.set_ylabel('accuracy')
    lns3 = ax2.plot(history['val_accuracy'], label='val_accuracy', color='green')

    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')

    ax1.grid(axis='x')
    ax2.grid(axis='y')

    # lims
    ax1.set_xlim((0, len(history['batch_loss'])))
    if loss_lim is not None:
        ax1.set_ylim(loss_lim)
    if accuracy_lim is not None:
        ax2.set_ylim(accuracy_lim)
    ax1.set_title(title)

    # saving / display
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def prettify_info_files(dir: str, indent: int = 2):
    dir = Path(dir)
    for file in dir.iterdir():
        if file.suffix != '.json':
            continue
        with open(file, 'r') as f:
            content = json.load(f)
        with open(file, 'w') as f:
            json.dump(content, f, indent=indent)

def save_train_history_figures_info_dir(dir: str):
    dir = Path(dir)
    for info_path in [p for p in dir.iterdir() if p.suffix == '.json']:
        save_path = info_path.with_suffix('.png')
        create_train_history_figure(
            info_path=str(info_path),
            save_path=str(save_path)
        )

def get_params_from_name(filename: str):
    params = {}

    for type in ['raw', 'event', 'joint']:
        if f'{type}.' in filename:
            params['DATA_TYPE'] = type

    if params['DATA_TYPE'] in ['raw', 'joint']:
        res = re.match(r'.*\.rawmax(\d+)\..*', filename)
        params['RAW_MAX_LEN'] = int(res.group(1))
    if params['DATA_TYPE'] in ['event', 'joint']:
        res = re.match(r'.*\.evmax(\d+)\..*', filename)
        params['EVENT_MAX_LEN'] = int(res.group(1))

    res = re.match(r'.*\.u(\d+)\..*', filename)
    params['UNITS'] = int(res.group(1))

    res = re.match(r'.*\.b(\d+)\..*', filename)
    params['BATCH_SIZE'] = int(res.group(1))

    res = re.match(r'.*\.ep(\d+)\..*', filename)
    params['EPOCHS'] = int(res.group(1))

    res = re.match(r'.*\.pat(\d+)\..*', filename)
    params['PATIENCE'] = int(res.group(1))

    res = re.match(r'.*\.tf(\d)\..*', filename)
    params['TEACHER_FORCING'] = False # bool(res.group(1))

    res = re.match(r'.*\.ed(\d)\..*', filename)
    params['EVENT_DETECTION'] = bool(res.group(1))

    res = re.match(r'.*\.emb(\d)\..*', filename)
    if res:
        params['EMBEDDING_DIM'] = int(res.group(1))
    else:
        params['EMBEDDING_DIM'] = 1

    for rnn_type in ['gru', 'lstm', 'bigru', 'bilstm']:
        if f'{rnn_type}.' in filename:
            params['RNN_TYPE'] = rnn_type

    for att_type in ['bahdanau', 'luong']:
        if f'{att_type}.' in filename:
            params['ATTENTION_TYPE'] = att_type
    if 'ATTENTION_TYPE' not in params:
        params['ATTENTION_TYPE'] = 'bahdanau'

    return params

def plot_attention_weights_for_prediction(model_path, input_data, save_path: str = None, seq_id: int = 0, output_max_length=50, figsize=(6, 2)):
    params = get_params_from_name(model_path)
    print(params)

    dm = DataModule(
        dir='data/simulator/random_200k_perfect',
        max_raw_length=0,
        max_event_length=0,
        bases_offset=0,
        batch_size=0,
        load_source=0,
        random_seed=0,
        verbose=True
    )

    basecaller = Basecaller(
        units=params['UNITS'],
        output_text_processor=dm.output_text_processor,
        input_data_type=params['DATA_TYPE'],
        input_padding_value=dm.input_padding_value,
        rnn_type=params['RNN_TYPE'],
        teacher_forcing=params['TEACHER_FORCING'],
        attention_type=params['ATTENTION_TYPE'],
        embedding_dim=params['EMBEDDING_DIM']
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(),
        loss=utils.MaskedLoss(basecaller.output_padding_token),
    )

    basecaller.load_weights(model_path)

    basecall_tokens_res = basecaller.tf_basecall_batch_to_tokens(input_data, output_max_length=output_max_length, early_break=True)
    att = basecall_tokens_res['attention'][seq_id]

    fig, ax = plt.subplots(figsize=figsize)

    ax.matshow(att, cmap='viridis', vmin=0.0)

    ax.set_xlabel('Encoder outputs id')
    ax.set_ylabel('Output bases id')
    model_details = model_path.replace('models/', '').replace('/model_chp', '')
    # plt.suptitle(f'Attention weights\n{model_details}')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
