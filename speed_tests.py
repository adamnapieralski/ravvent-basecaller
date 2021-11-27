from run_simulator_reduced import RANDOM_SEED
import tensorflow as tf
from timeit import default_timer as timer
import json

from data_loader import DataModule
from basecaller import Basecaller
import utils
from pathlib import Path

RANDOM_SEED = 22

def run_speed_test_chiron(multiple_seqs_dir):
    dir = Path(multiple_seqs_dir)
    single_seq_path = [p for p in dir.iterdir()]
    single_seq_path.sort()

    for seq_path in single_seq_path:
        id = seq_path.as_posix()[-2:]

        start_time = timer()

        dm = DataModule(
            dir=single_seq_path[0],
            max_raw_length=200,
            max_event_length=30,
            bases_offset=6,
            batch_size=128,
            train_size=1,
            val_size=0,
            test_size=0,
            load_source='chiron',
            event_detection=True,
            random_seed=RANDOM_SEED,
            verbose=False
        )
        dm.setup()

        data_loading_time = timer() - start_time

        train_ds = dm.dataset_train

        basecaller = Basecaller(
            units=128,
            output_text_processor=dm.output_text_processor,
            input_data_type='event',
            input_padding_value=dm.input_padding_value,
            embedding_dim=5,
            rnn_type='bilstm',
            teacher_forcing=False,
            attention_type='bahdanau'
        )

        # checkpoint_filepath = 'models/simulator.reduced/model.1.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.4096.600000.4096.81/model_chp'
        checkpoint_filepath = 'models/chiron/model.event.bilstm.u128.chiron.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.05/model_chp'

        basecaller.load_weights(checkpoint_filepath)

        start_time = timer()
        basecaller.basecall_data(train_ds)
        basecalling_time = timer() - start_time
        print("{},{},{}".format(id,data_loading_time, basecalling_time))


if __name__ == "__main__":
    run_speed_test_chiron('data/chiron/eval/ecoli_eval_0001_0080_single')