# ravvent-basecaller

Basecaller, called Ravvent, using joint raw and event data sequence-to-sequence processing. Incorporating an encoder-decoder architecture with attention mechanism and LSTMs as RNNs.

## Structure
- [data_loader.py](data_loader.py) - includes DataGenerator class responsible for loading simulated and real data, then preprocessing it into final batches, consumable by model
- [basecaller.py](basecaller.py) - includes Encoder, Decoder classes, and general Basecaller class, combining all layers into single keras Model responsible for learning and performing basecalling task
- [utils.py](utils.py) - includes various utilities functions
- [ravvent.py](ravvent.py) - sample script for running learning model pipeline
- [ravvent_mapping_evaluator.py](ravvent_mapping_evaluator.py), [ravvent_performance_evaluator.py](ravvent_performance_evaluator.py) - scripts for performing evaluation (read accuracy, speed)

## Data availability
### Simulated
Simulated datasets were generated using [DeepSimulator](https://github.com/liyu95/DeepSimulator) tool. Script used in this purpose is [generate_simulator_reduced.py](data/generate_simulator_reduced.py), where the parameters for execution of DeepSimulator, as well as event_detection, can be found. _Fasta_ files are stored in [data/simulator/reduced](data/simulator/reduced) directory.

### Real
Real data source is supporting data for [Chiron](https://github.com/haotianteng/Chiron) basecaller, that is available [here](http://gigadb.org/dataset/100425).

## Environment
### Prerequisites
- tensorflow >= 2.7
- [ont_fast5_api](https://github.com/nanoporetech/ont_fast5_api)
- numpy