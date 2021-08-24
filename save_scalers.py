import tensorflow as tf

from data_loader import DataModule

RANDOM_SEED = 22

def save_fit_scalers(
    data_string: str,
    batch_size: int,
    raw_max_len: int,
    event_max_len: int,
    bases_offset: int,
    event_detection: bool
):
    tf.random.set_seed(RANDOM_SEED)

    load_source = 'simulator'

    dm = DataModule(
        dir='data/simulator/reduced/{}.train'.format(data_string),
        max_raw_length=raw_max_len,
        max_event_length=event_max_len,
        bases_offset=bases_offset,
        batch_size=batch_size,
        train_size=1,
        val_size=0,
        test_size=0,
        load_source=load_source,
        event_detection=event_detection,
        random_seed=RANDOM_SEED,
        verbose=True
    )
    dm.setup()

    scaler_path = f'data/simulator/reduced/scalers/scalers.{data_string}.rawmax{raw_max_len}.evmax{event_max_len}.off{bases_offset}.pkl'
    dm.save_scalers(scaler_path)

if __name__ == '__main__':
    basic_params = {
        'data_string': '',
        'batch_size': 128,
        'raw_max_len': 200,
        'event_max_len': 30,
        'bases_offset': 1,
        'event_detection': True
    }

    variants = [
        {'data_string': 'seq.3.10000.45', 'event_detection': True},
        {'data_string': 'seq.12.75000.450', 'event_detection': True},
        {'data_string': 'seq.21.150000.1024', 'event_detection': True},
        {'data_string': 'seq.43.300000.2048', 'event_detection': True},
        {'data_string': 'seq.4096.600000.4096', 'event_detection': True},
    ]

    for v in variants:
        save_fit_scalers(
            **{**basic_params, **v}
        )