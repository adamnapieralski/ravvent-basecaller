import tensorflow as tf
from timeit import default_timer as timer
import json

from data_loader import DataModule
from basecaller import Basecaller
import utils

RANDOM_SEED = 22

def run_benchmark(
    data_source_string: str,
    data_test_string: str,
    model_path: str,
    data_type: str,
    batch_size: int,
    raw_max_len: int,
    event_max_len: int,
    units: int,
    epochs: int,
    patience: int,
    bases_offset: int,
    rnn_type: str,
    attention_type: str,
    embedding_dim: int,
    event_detection: bool
):
    tf.random.set_seed(RANDOM_SEED)

    load_source = 'simulator'
    teacher_forcing = False

    if data_type == 'joint':
        name_max_len = f'rawmax{raw_max_len}.evmax{event_max_len}'
    else:
        name_max_len = f'rawmax{raw_max_len}' if data_type == 'raw' else f'evmax{event_max_len}'

    name_spec = f'{data_type}.{rnn_type}.u{units}.{load_source}.{name_max_len}.b{batch_size}.off{bases_offset}.ep{epochs}.pat{patience}.tf{int(teacher_forcing)}.emb{embedding_dim}.ed{int(event_detection)}.{attention_type}'

    name = name_spec + '.reduced.' + data_source_string
    print('RUNNING', name)

    name = f'{data_test_string}.{data_type}.off{bases_offset}'

    dm = DataModule(
        dir='data/simulator/reduced/{}.bcall'.format(data_test_string),
        max_raw_length=raw_max_len,
        max_event_length=event_max_len,
        bases_offset=bases_offset,
        batch_size=batch_size,
        train_size=0,
        val_size=0,
        test_size=1,
        load_source=load_source,
        event_detection=event_detection,
        random_seed=RANDOM_SEED,
        shuffle=False, # crucial
        verbose=True
    )
    scalers_path = f'data/simulator/reduced/scalers/scalers.{data_source_string}.rawmax200.evmax30.off1.pkl'
    dm.load_scalers(scalers_path)
    dm.setup()
    test_ds = dm.dataset_test

    basecaller = Basecaller(
        units=units,
        output_text_processor=dm.output_text_processor,
        input_data_type=data_type,
        input_padding_value=dm.input_padding_value,
        embedding_dim=embedding_dim,
        rnn_type=rnn_type,
        teacher_forcing=teacher_forcing,
        attention_type=attention_type
    )

    # Configure the loss and optimizer
    basecaller.compile(
        optimizer=tf.optimizers.Adam(),
        loss=utils.MaskedLoss(basecaller.output_padding_token),
    )

    basecaller.load_weights(model_path)

    acc = basecaller.evaluate_basecall_full_aligned(
        test_ds,
        f'{dm.dir}/sampled_read.fasta',
        f'basecalled.{name}.fastq'
    )

    info = { 'accuracy': acc }

    with open(f'basecalled.acc.{name}.json', 'w') as info_file:
        json.dump(info, info_file, indent=2)

if __name__ == '__main__':
    basic_params = {
        'data_source_string': '',
        'data_test_string': '',
        'model_path': '',
        'data_type': '',
        'batch_size': 128,
        'raw_max_len': 200,
        'event_max_len': 30,
        'units': 128,
        'epochs': 100,
        'patience': 100,
        'bases_offset': 8,
        'rnn_type': 'bilstm',
        'attention_type': 'bahdanau',
        'embedding_dim': 5,
        'event_detection': True
    }

    variants = [
        {'data_source_string': 'seq.4096.600000.4096', 'data_test_string': 'seq.4096.120000.4096', 'data_type': 'joint', 'model_path': 'models/simulator.reduced/model.1.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.4096.600000.4096.81/model_chp'},
        {'data_source_string': 'seq.4096.600000.4096', 'data_test_string': 'seq.4096.120000.4096', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.4096.600000.4096/model_chp'},
        # {'data_source_string': 'seq.43.300000.2048', 'data_test_string': 'seq.43.60000.2048', 'data_type': 'joint', 'model_path': 'models/simulator.reduced/model.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.43.300000.2048/model_chp'},
        # {'data_source_string': 'seq.43.300000.2048', 'data_test_string': 'seq.43.60000.2048', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.43.300000.2048/model_chp'},
        # {'data_source_string': 'seq.21.150000.1024', 'data_test_string': 'seq.21.30000.1024', 'data_type': 'joint', 'model_path': 'models/simulator.reduced/model.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.21.150000.1024/model_chp'},
        # {'data_source_string': 'seq.21.150000.1024', 'data_test_string': 'seq.21.30000.1024', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.21.150000.1024/model_chp'},
        # {'data_source_string': 'seq.12.75000.450', 'data_test_string': 'seq.12.15000.450', 'data_type': 'joint', 'model_path': 'models/simulator.reduced/model.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.12.75000.450/model_chp'},
        # {'data_source_string': 'seq.12.75000.450', 'data_test_string': 'seq.12.15000.450', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.12.75000.450/model_chp'},
        # {'data_source_string': 'seq.3.25000.45', 'data_test_string': 'seq.3.5000.45', 'data_type': 'joint', 'model_path': 'models/simulator.reduced/model.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.3.25000.45/model_chp'},
        # {'data_source_string': 'seq.3.25000.45', 'data_test_string': 'seq.3.5000.45', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.3.25000.45/model_chp'},




        # {'data_source_string': 'seq.12.75000.450', 'data_test_string': 'seq.12.15000.450', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.12.75000.450/model_chp'},
        # {'data_source_string': 'seq.43.300000.2048', 'data_test_string': 'seq.43.60000.2048', 'data_type': 'raw', 'model_path': 'models/simulator.reduced/model.raw.bilstm.u128.simulator.rawmax200.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.43.300000.2048/model_chp'},
        # {'data_source_string': 'seq.43.300000.2048', 'data_test_string': 'seq.43.60000.2048', 'data_type': 'joint', 'model_path': 'models/simulator.reduced/model.joint.bilstm.u128.simulator.rawmax200.evmax30.b128.ep100.pat100.tf0.emb5.ed1.bahdanau.reduced.seq.43.300000.2048/model_chp'},
        # {'data_source_string': 'seq.3.25000.45', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.3.25000.45', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.12.75000.450', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.12.75000.450', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.12.75000.450', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.21.150000.1024', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.21.150000.1024', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.21.150000.1024', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.43.300000.2048', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.43.300000.2048', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.43.300000.2048', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.4096.600000.4096', 'data_type': 'raw', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.4096.600000.4096', 'data_type': 'event', 'rnn_type': 'bilstm', 'event_detection': True},
        # {'data_source_string': 'seq.4096.600000.4096', 'data_type': 'joint', 'rnn_type': 'bilstm', 'event_detection': True},
    ]

    for v in variants:
        run_benchmark(
            **{**basic_params, **v}
        )