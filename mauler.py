import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sys

from shape_checker import ShapeChecker

import mauler_data as md

import typing
from typing import Any, Tuple

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
                tf.keras.layers.GRUCell(
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

    return output, tf.concat(states, axis=-1)



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, dec_units, batch_sz, max_input_len, max_output_len, attention_type='bahdanau', teacher_forcing=True):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type

    self.max_input_len = max_input_len
    self.max_output_len = max_output_len
    self.teacher_forcing = teacher_forcing

    self.vocab_size = vocab_size

    # Embedding Layer
    self.embedding = lambda ids: tf.one_hot(ids, depth=self.vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.GRUCell(self.dec_units)

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
    self.rnn_cell = self.build_rnn_cell()

    # Define the decoder with respect to fundamental rnn cell
    if teacher_forcing:
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
    else:
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc, maximum_iterations=max_output_len - 1)

  def build_rnn_cell(self):
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
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
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

    def __init__(self, enc_units: int, dec_units: int, batch_sz: int, encoder_depth: int = 2, teacher_forcing: bool = True, attention_type: str = 'luong'):
        super().__init__()
        # Build the encoder and decoder
        self.batch_sz = batch_sz
        self.encoder = Encoder(enc_units, batch_sz, encoder_depth)
        self.max_output_len = 42
        self.decoder = Decoder(
            vocab_size=5,
            dec_units=dec_units,
            batch_sz=batch_sz,
            max_input_len=300,
            max_output_len=40,
            attention_type='luong',
            teacher_forcing=teacher_forcing
        )

        self.teacher_forcing = teacher_forcing
        self.attention_type = attention_type
        self.shape_checker = ShapeChecker()

        # print('Input padding', self.input_padding_value)



        self.output_start_token = np.int32(md.start_token)
        self.output_end_token = np.int32(md.end_token)


    ##
    ## TRAINING
    ##
    def loss_function(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real, self.output_end_token))   #output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_sum(loss)
        loss /= tf.reduce_sum(mask)
        return loss

    def train_step(self, inputs):
        return self._tf_train_step(inputs)

    def _train_step(self, data):
        (raw, target) = data

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(raw, training=True)

            loss = tf.constant(0.0)

            dec_input = target[:, :-1]
            real = target[:, 1:]

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

    def _val_step(self, data, return_outputs=False):
        (raw, target) = data

        enc_output, enc_state = self.encoder(raw, training=True)

        start_tokens = tf.fill([self.batch_sz], self.output_start_token)

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(self.decoder.embedding)

        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc, maximum_iterations=self.max_output_len - 1)

        self.decoder.attention_mechanism.setup_memory(enc_output)
        decoder_initial_state = self.decoder.build_initial_state(self.batch_sz, enc_state, tf.float32)

        outputs, _, _ = decoder_instance(None, start_tokens=start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)

        logits = outputs.rnn_output

        logits = tf.pad(logits, [[0,0], [0, self.max_output_len - 1 - tf.shape(logits)[1]], [0,0]])

        loss = self.loss_function(target[:,1:], logits)

        if return_outputs:
            return {'loss': loss}, outputs

        return {'loss': loss}

    @tf.function
    def _tf_val_step(self, inputs):
        return self._val_step(inputs)

    ##
    ## EVALUATION - TEST
    ##

    def beam_search_prediction(self, data, beam_width):
        (raw, target) = data
        enc_output, enc_state = self.encoder(raw, training=False)
        start_tokens = tf.fill([self.batch_sz], self.output_start_token)
        enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_width)
        enc_state = tfa.seq2seq.tile_batch(enc_state, multiplier=beam_width)

        self.decoder.attention_mechanism.setup_memory(enc_output)

        decoder_initial_state = self.decoder.build_initial_state(beam_width * self.batch_sz, enc_state, tf.float32)
        # decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width * batch_size, dtype=tf.float32)
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(
            cell=self.decoder.rnn_cell,
            beam_width=beam_width,
            embedding_fn=self.decoder.embedding,
            output_layer=self.decoder.fc,
            maximum_iterations=self.max_output_len - 1
        )
        outputs, _, _ = decoder_instance(None, start_tokens=start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)

        return outputs.predicted_ids[:,:,0], outputs.beam_search_decoder_output.scores[:,:,0]


    def beam_evaluate(self, data, beam_width=3):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

        enc_output, enc_state, batch_size = self._encode_input(input_data, training=False)

        start_tokens = tf.fill([self.batch_sz], self.output_start_token)

        enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_width)

        self.decoder.attention_mechanism.setup_memory(enc_output)

        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*self.batch_sz, dtype=tf.float32)
        # decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

        # Instantiate BeamSearchDecoder
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(
            cell=self.decoder.rnn_cell,
            beam_width=beam_width,
            embedding_fn=self.decoder.embedding,
            output_layer=self.decoder.fc,
        )
        outputs, _, _ = decoder_instance(None, start_tokens=start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)

        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
        beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

        return final_outputs.numpy(), beam_scores.numpy()

    def harsh_accuracy(self, data, beam_width):
        (raw, target) = data
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
