import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sys

from shape_checker import ShapeChecker

import typing
from typing import Any, Tuple

from Bio import pairwise2

import utils

class Encoder(tf.keras.Model):
  def __init__(self, enc_units, batch_sz, layer_depth):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units

    self.layer_depth = layer_depth

    self.bidir_layers = [
        tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(
                tf.keras.layers.LSTMCell(
                    enc_units, kernel_initializer='glorot_uniform',
                ),
                return_sequences=True, return_state=True,
            )
        ) for _ in range(self.layer_depth)
    ]

  def call(self, inputs, training=False):
    inputs.set_shape((None, None, 1))
    output = inputs
    states = None
    for i in range(self.layer_depth):
        result = self.bidir_layers[i](output, initial_state=states, training=training)
        output, states = result[0], result[1:]

    return output, states



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, layer_depth, dec_units, batch_sz, max_input_len, max_output_len, attention_type='bahdanau', teacher_forcing=True):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type

    self.max_input_len = max_input_len
    self.max_output_len = max_output_len
    self.teacher_forcing = teacher_forcing

    self.vocab_size = vocab_size

    self.layer_depth = layer_depth

    # Embedding Layer
    # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.embedding = lambda ids: tf.one_hot(ids, depth=self.vocab_size)

    cells = [tf.keras.layers.LSTMCell(dec_units, kernel_initializer='glorot_uniform') for _ in range(self.layer_depth)]

    self.decoder_rnn_cell = tf.keras.layers.StackedRNNCells(cells)


    # Define the fundamental cell for decoder recurrent structure
    # self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Sampler
    if type(teacher_forcing) is bool:
        if teacher_forcing:
            print('TrainingSampler')
            self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        else:
            print('GreedyEmbeddingSampler')
            self.sampler = tfa.seq2seq.GreedyEmbeddingSampler(
                self.embedding
            )
    else:
        print('ScheduledEmbeddingTrainingSampler')
        self.sampler = tfa.seq2seq.ScheduledEmbeddingTrainingSampler(
            teacher_forcing,
            self.embedding
            )

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                              None, self.batch_sz*[self.max_input_len], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    if teacher_forcing:
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
    else:
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc, maximum_iterations=max_output_len - 1)

  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    # decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state=None, start_tokens=None, end_token=None):
    if type(self.teacher_forcing) is bool:
        if self.teacher_forcing:
            x = self.embedding(inputs)
            outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_output_len-1], training=True)
        else:
            outputs, _, _ = self.decoder(None, start_tokens=start_tokens, end_token=end_token, initial_state=initial_state, training=True)
    else:
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_output_len-1], training=True)
    return outputs



class Basecaller(tf.keras.Model):

    def __init__(self, enc_units: int, dec_units: int, batch_sz: int, output_text_processor, input_data_type: str, input_padding_value, encoder_depth: int = 3, decoder_depth: int = 3, rnn_type: str = 'gru', teacher_forcing: bool = True, attention_type: str = 'bahdanau', beam_width: int = 5):
        """Initialize

        Args:
            units (int): Number of units (encoders and decoder)
            output_text_processor ([type]): From DataLoader
            input_data_type (str): {'raw', 'event', 'joint'}
            input_padding_value ([type]): From DataLoader
            rnn_type (str, optional): Type of RNN to use, {'gru', 'lstm', 'bigru', 'bilstm'}. Defaults to 'gru'.
            teacher_forcing (bool, optional): If use teacher forcing during training. Defaults to True.
            attention_type (str, optional): {'bahdanau', 'luong'}. Defaults to 'bahdanau'.
        """
        super().__init__()
        # Build the encoder and decoder
        self.batch_sz = batch_sz
        self.encoder_raw = Encoder(enc_units, batch_sz, encoder_depth)
        self.encoder_event = None
        self.max_output_len = 40
        self.decoder = Decoder(
            vocab_size=len(output_text_processor.get_vocabulary()),
            layer_depth=decoder_depth,
            dec_units=dec_units,
            batch_sz=batch_sz,
            max_input_len=200,
            max_output_len=self.max_output_len,
            attention_type='luong',
            teacher_forcing=teacher_forcing
        )
        output_text_processor._output_sequence_length = self.max_output_len

        self.output_text_processor = output_text_processor
        self.input_data_type = input_data_type
        self.input_padding_value = input_padding_value
        self.rnn_type = rnn_type
        self.teacher_forcing = teacher_forcing
        self.attention_type = attention_type
        self.beam_width = beam_width

        self.shape_checker = ShapeChecker()

        self.output_token_string_from_index = (
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True
            )
        )
        index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(),
            mask_token=''
        )
        self.output_start_token = np.int32(index_from_string('[START]'))
        self.output_end_token = np.int32(index_from_string('[END]'))
        self.output_padding_token = index_from_string('')

        # The test output should never generate padding, unknown, or start.
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()
        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.test_token_mask = token_mask


    ##
    ## TRAINING
    ##
    def loss_function(self, real, pred):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real, self.output_padding_token))   #output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_sum(loss)
        loss /= tf.reduce_sum(mask)
        return loss

    def train_step(self, inputs):
        return self._tf_train_step(inputs)

    def _train_step(self, data):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

        with tf.GradientTape() as tape:
            enc_output, enc_state, _ = self._encode_input(input_data, training=True)

            loss = tf.constant(0.0)

            dec_input = target_tokens[:, :-1]
            real = target_tokens[:, 1:]

            self.decoder.attention_mechanism.setup_memory(enc_output)

            decoder_initial_state = self.decoder.build_initial_state(self.batch_sz, enc_state, tf.float32)

            if self.teacher_forcing:
                pred = self.decoder(dec_input, initial_state=decoder_initial_state)
                logits = pred.rnn_output
            else:
                start_tokens = tf.fill([self.batch_sz], self.output_start_token)
                pred = self.decoder(None, initial_state=decoder_initial_state, start_tokens=start_tokens, end_token=self.output_end_token)
                logits = pred.rnn_output
                logits = tf.pad(logits, [[0,0], [0, self.max_output_len - 1 - tf.shape(logits)[1]], [0,0]])

            loss = self.loss_function(real, logits)

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)

        # # CLIPPTING
        # gradients = [tf.clip_by_norm(g, self.grad_clip_norm) for g in gradients]

        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'loss': loss}

    @tf.function
    def _tf_train_step(self, data):
        return self._train_step(data)


    ##
    ## VALIDATION
    ##

    def test_step(self, inputs):
        return self._tf_val_step(inputs)

    def _val_step(self, data):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

        pred_tokens_greedy, logits_greedy = self.greedy_search_prediction(input_data)

        logits_greedy = tf.pad(logits_greedy, [[0,0], [0, self.max_output_len - 1 - tf.shape(logits_greedy)[1]], [0,0]])
        loss = self.loss_function(target_tokens[:,1:], logits_greedy)

        pred_tokens_greedy = tf.pad(pred_tokens_greedy, [[0,0], [0, self.max_output_len - 1 - tf.shape(pred_tokens_greedy)[1]]])
        align_acc_greedy = self.pairwise_accuracy(tf.strings.regex_replace(target_sequence, '\s', ''), self.tokens_to_bases_sequence(tf.cast(pred_tokens_greedy, tf.int64)))
        harsh_acc_greedy = utils.masked_accuracy(target_tokens[:,1:], tf.cast(pred_tokens_greedy, tf.int64), [self.output_padding_token, self.output_start_token, self.output_end_token])

        # pred_tokens_beam, _ = self.beam_search_prediction(input_data, self.beam_width)
        # pred_tokens_beam = tf.pad(pred_tokens_beam, [[0,0], [0, self.max_output_len - 1 - tf.shape(pred_tokens_beam)[1]]])
        # align_acc_beam = self.pairwise_accuracy(tf.strings.regex_replace(target_sequence, '\s', ''), self.tokens_to_bases_sequence(tf.cast(pred_tokens_beam, tf.int64)))
        # harsh_acc_beam = utils.masked_accuracy(target_tokens[:,1:], tf.cast(pred_tokens_beam, tf.int64), [self.output_padding_token, self.output_start_token, self.output_end_token])

        return {'loss': loss, 'align_acc_greedy': align_acc_greedy, 'harsh_acc_greedy': harsh_acc_greedy} #, 'align_acc_beam': align_acc_beam, 'harsh_acc_beam': harsh_acc_beam}

    # @tf.function
    def _tf_val_step(self, inputs):
        return self._val_step(inputs)

    ##
    ## EVALUATION - TEST
    ##

    def tokens_to_bases_sequence(self, result_tokens):
        result_core_tokens = tf.where(
            tf.math.logical_or(
                tf.math.equal(result_tokens, self.output_end_token), tf.math.equal(result_tokens, self.output_start_token)),
                tf.fill(tf.shape(result_tokens), self.output_padding_token), result_tokens)
        result_text_tokens = self.output_token_string_from_index(result_core_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator='')
        result_text = tf.strings.upper(result_text)
        return result_text

    def beam_search_prediction(self, input_data, beam_width):
        enc_output, _, batch_size = self._encode_input(input_data, training=False)
        start_tokens = tf.fill([batch_size], self.output_start_token)
        enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_width)

        self.decoder.attention_mechanism.setup_memory(enc_output)

        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*batch_size, dtype=tf.float32)
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(
            cell=self.decoder.rnn_cell,
            beam_width=beam_width,
            embedding_fn=self.decoder.embedding,
            output_layer=self.decoder.fc,
            maximum_iterations=self.max_output_len - 1
        )
        outputs, _, _ = decoder_instance(None, start_tokens=start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)

        return outputs.predicted_ids[:,:,0], outputs.beam_search_decoder_output.scores[:,:,0]

    def greedy_search_prediction(self, input_data):
        enc_output, enc_state, batch_size = self._encode_input(input_data, training=False)

        start_tokens = tf.fill([batch_size], self.output_start_token)

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(self.decoder.embedding)

        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc, maximum_iterations=self.max_output_len - 1)

        self.decoder.attention_mechanism.setup_memory(enc_output)
        decoder_initial_state = self.decoder.build_initial_state(batch_size, enc_state, tf.float32)

        outputs, _, _ = decoder_instance(None, start_tokens = start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)
        return outputs.sample_id, outputs.rnn_output

    def harsh_accuracy(self, data, beam_width):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)
        (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

        result, _ = self.beam_evaluate(data, beam_width=beam_width)
        result = result[:,0]
        real = target_tokens[:, 1:]

        result_shape = tf.shape(result)
        if result_shape[1] > self.max_output_len - 1:
            result = tf.slice(result, [0, 0], [result_shape[0], self.max_output_len - 1])
        elif result_shape[1] < self.max_output_len - 1:
            result = tf.pad(result, [[0,0], [0, self.max_output_len - 1 - result_shape[1]]])

        acc = utils.masked_accuracy(real, result, [self.output_padding_token, self.output_start_token, self.output_end_token])

        return acc

    def pairwise_accuracy(self, target_sequences, pred_sequences, match_sc=1., mismatch_sc=-1., gap_open_sc=-1., gap_extend_sc=-0.2):
        target_seqs = [s.numpy().decode('UTF-8') for s in target_sequences]
        pred_seqs = [s.numpy().decode('UTF-8') for s in pred_sequences]
        assert(len(target_seqs) == len(pred_seqs))

        sc_sum = 0
        len_sum = 0
        for pred, targ in zip(pred_seqs, target_seqs):
            algn = pairwise2.align.globalms(pred, targ, match_sc, mismatch_sc, gap_open_sc, gap_extend_sc)
            if len(algn) >= 1:
                sc_sum += algn[0].score
            len_sum += len(targ)
        if len_sum > 0:
            return sc_sum / len_sum
        return 0


    # def basecall_merge(self, data, beam_width):
    #     input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)
    #     (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

    #     result, _ = self.beam_evaluate(data, beam_width=beam_width)


    ##
    ## GENERAL
    ##

    def _preprocess(self, input_data, target_sequence):
        input_mask = self._prepare_input_mask(input_data)
        target_tokens = self.output_text_processor(target_sequence)
        target_mask = target_tokens != self.output_padding_token

        return input_data, input_mask, target_tokens, target_mask


    def _prepare_joint_encoder_state(self, enc_state_raw, enc_state_event):
        # when state is a list of different states (like in lstm or bidirectional rnns)
        if len(tf.shape(enc_state_raw)) == 3:
            enc_state = []
            for st_raw, st_event in zip(enc_state_raw, enc_state_event):
                enc_state.append(tf.add(st_raw, st_event))
        else:
            enc_state = tf.add(enc_state_raw, enc_state_event)
        return enc_state

    def _prepare_input_mask(self, input_data):
        if self.input_data_type == 'joint':
            (raw_input, event_input) = input_data
            input_mask_raw = utils.input_mask(raw_input, self.input_padding_value)
            input_mask_event = utils.input_mask(event_input, self.input_padding_value)
            input_mask = tf.concat((input_mask_raw, input_mask_event), axis=-1)
        else:
            input_mask = utils.input_mask(input_data, self.input_padding_value)

        return input_mask

    def _encode_input(self, input_data, return_mask=False, training=False):
        if self.input_data_type == 'joint':
            (raw_input, event_input) = input_data

            enc_output_raw, enc_state_raw = self.encoder_raw(raw_input, training=training)

            enc_output_event, enc_state_event = self.encoder_event(event_input, training=training)

            enc_output = tf.concat((enc_output_raw, enc_output_event), axis=1)
            enc_state = self._prepare_joint_encoder_state(enc_state_raw, enc_state_event)

            batch_size = tf.shape(raw_input)[0]

        elif self.input_data_type == 'raw':
            enc_output, enc_state = self.encoder_raw(input_data, training=training)
            batch_size = tf.shape(input_data)[0]

        elif self.input_data_type == 'event':
            enc_output, enc_state= self.encoder_event(input_data, training=training)
            # self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            # self.shape_checker(enc_state, ('batch', 'enc_units'))
            batch_size = tf.shape(input_data)[0]

        if return_mask:
            input_mask = self._prepare_input_mask(input_data)
            return enc_output, enc_state, batch_size, input_mask

        return enc_output, enc_state, batch_size
