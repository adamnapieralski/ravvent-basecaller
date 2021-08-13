"""
Various utils functions for output analysis
"""
import matplotlib.pyplot as plt
import json, re
from pathlib import Path
from typing import Tuple

def create_train_history_figure(info_path: str, save_path: str = None, loss_lim: Tuple[float, float] = None, accuracy_lim: Tuple[float, float] = (0, 0.6), figsize: Tuple[int,int] = (6,4)):
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
