import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import utils

class Encoder(tf.keras.Model):
  def __init__(self, enc_units, batch_sz, layer_depth, inputs_features_num):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units

    self.layer_depth = layer_depth
    self.inputs_features_num = inputs_features_num

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

  def call(self, inputs, training=False, mask=None):
    inputs.set_shape((None, None, self.inputs_features_num))
    output = inputs
    states = None
    for i in range(self.layer_depth):
        kwargs = {'initial_state': states, 'training': training}
        if mask is not None:
            kwargs['mask'] = mask
        result = self.bidir_layers[i](output, **kwargs)
        output, states = result[0], result[1:]

    return output, states



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, layer_depth, dec_units, batch_sz, max_input_len, attention_type='luong', teacher_forcing=True):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type

    self.max_input_len = max_input_len

    if teacher_forcing is False:
        self.teacher_forcing = 1.
    else:
        self.teacher_forcing = teacher_forcing

    self.vocab_size = vocab_size

    self.layer_depth = layer_depth

    # Embedding Layer
    self.embedding = lambda ids: tf.one_hot(ids, depth=self.vocab_size)

    cells = [tf.keras.layers.LSTMCell(dec_units, kernel_initializer='glorot_uniform') for _ in range(self.layer_depth)]

    self.decoder_rnn_cell = tf.keras.layers.StackedRNNCells(cells)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Sampler
    if type(self.teacher_forcing) is bool:
        if teacher_forcing:
            print('TrainingSampler')
            self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        else:
            raise NotImplementedError()
    else:
        print('ScheduledEmbeddingTrainingSampler')
        self.sampler = tfa.seq2seq.ScheduledEmbeddingTrainingSampler(
            self.teacher_forcing,
            self.embedding)

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                              None, self.batch_sz*[self.max_input_len], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

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


  def call(self, inputs, max_output_len=None, initial_state=None, start_tokens=None, end_token=None):
    if type(self.teacher_forcing) is bool:
        if self.teacher_forcing:
            x = self.embedding(inputs)
            outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_output_len-1], training=True)
        else:
            raise NotImplementedError()
    else:
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_output_len-1], training=True)
    return outputs



class Basecaller(tf.keras.Model):

    def __init__(self, enc_units: int, dec_units: int, batch_sz: int, tokenizer: tf.keras.preprocessing.text.Tokenizer, input_data_type: str, input_padding_value, encoder_depth: int = 2, decoder_depth: int = 1, rnn_type: str = 'bilstm', teacher_forcing: bool = True, attention_type: str = 'luong', beam_width: int = 5):
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
        self.encoder_raw = Encoder(enc_units, batch_sz, encoder_depth, 1)
        self.encoder_event = Encoder(enc_units, batch_sz, encoder_depth, 5)

        self.tokenizer = tokenizer

        if input_data_type == 'raw':
            self.max_input_len = 200
        elif input_data_type == 'event':
            self.max_input_len = 30
        else:
            self.max_input_len = 230

        self.teacher_forcing = teacher_forcing
        self.decoder = Decoder(
            vocab_size=len(tokenizer.word_index),
            layer_depth=decoder_depth,
            dec_units=dec_units,
            batch_sz=batch_sz,
            max_input_len=self.max_input_len,
            attention_type='luong',
            teacher_forcing=self.teacher_forcing
        )

        self.input_data_type = input_data_type
        self.input_padding_value = input_padding_value
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        self.beam_width = beam_width

        self.output_start_token = np.int32(tokenizer.word_index['$'])
        self.output_end_token = np.int32(tokenizer.word_index['^'])
        self.output_padding_token = np.int32(tokenizer.word_index[''])


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
        input_data, target_tokens = utils.unpack_data_to_input_target(data, self.input_data_type)
        max_output_len = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, input_mask = self._encode_input(input_data, training=True)

            loss = tf.constant(0.0)

            dec_input = target_tokens[:, :-1]
            real = target_tokens[:, 1:]

            self.decoder.attention_mechanism.setup_memory(enc_output, memory_mask=input_mask)

            decoder_initial_state = self.decoder.build_initial_state(self.batch_sz, None, tf.float32)

            pred = self.decoder(dec_input, max_output_len, initial_state=decoder_initial_state)
            logits = pred.rnn_output

            pred_tokens = pred.sample_id
            loss = self.loss_function(real, logits)

            acc = utils.masked_accuracy(real, tf.cast(pred_tokens, tf.int64), [self.output_padding_token, self.output_start_token, self.output_end_token])

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'loss': loss, 'acc': acc}

    @tf.function
    def _tf_train_step(self, data):
        return self._train_step(data)


    ##
    ## VALIDATION
    ##

    def test_step(self, inputs):
        return self._tf_val_step(inputs)

    def _val_step(self, data):
        input_data, target_tokens = utils.unpack_data_to_input_target(data, self.input_data_type)
        max_output_len = tf.shape(target_tokens)[1]

        pred_tokens_greedy, logits_greedy = self.greedy_search_prediction(input_data, max_output_len=max_output_len)

        logits_greedy = tf.pad(logits_greedy, [[0,0], [0, max_output_len - 1 - tf.shape(logits_greedy)[1]], [0,0]])
        loss = self.loss_function(target_tokens[:,1:], logits_greedy)

        pred_tokens_greedy = tf.pad(pred_tokens_greedy, [[0,0], [0, max_output_len - 1 - tf.shape(pred_tokens_greedy)[1]]])
        acc = utils.masked_accuracy(target_tokens[:,1:], tf.cast(pred_tokens_greedy, tf.int64), [self.output_start_token, self.output_end_token])

        return {'loss': loss, 'acc': acc}

    # @tf.function
    def _tf_val_step(self, inputs):
        return self._val_step(inputs)

    ##
    ## EVALUATION - TEST
    ##

    def tokens_to_nuc_sequences(self, result_tokens):
        result_text = self.tokenizer.sequences_to_texts(result_tokens.numpy())
        result_text = [
            rt.replace(' ', '').replace('^', '').replace('$', '').upper()
            for rt in result_text]
        return result_text

    def beam_search_prediction(self, input_data, beam_width, max_output_len):
        enc_output, input_mask = self._encode_input(input_data, training=False)
        batch_size = tf.shape(enc_output)[0]
        start_tokens = tf.fill([batch_size], self.output_start_token)
        enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_width)
        input_mask = tfa.seq2seq.tile_batch(input_mask, multiplier=beam_width)

        self.decoder.attention_mechanism.setup_memory(enc_output, memory_mask=input_mask)

        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*batch_size, dtype=tf.float32)
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(
            cell=self.decoder.rnn_cell,
            beam_width=beam_width,
            embedding_fn=self.decoder.embedding,
            output_layer=self.decoder.fc,
            maximum_iterations=max_output_len - 1
        )
        outputs, _, _ = decoder_instance(None, start_tokens=start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)

        return outputs.predicted_ids[:,:,0], outputs.beam_search_decoder_output.scores[:,:,0]

    def greedy_search_prediction(self, input_data, max_output_len):
        enc_output, input_mask = self._encode_input(input_data, training=False)
        batch_size = tf.shape(enc_output)[0]
        start_tokens = tf.fill([batch_size], self.output_start_token)

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(self.decoder.embedding)

        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc, maximum_iterations=max_output_len - 1)

        self.decoder.attention_mechanism.setup_memory(enc_output, memory_mask=input_mask)
        decoder_initial_state = self.decoder.build_initial_state(batch_size, None, tf.float32)

        outputs, _, _ = decoder_instance(None, start_tokens = start_tokens, end_token=self.output_end_token, initial_state=decoder_initial_state, training=False)
        return outputs.sample_id, outputs.rnn_output

    # def harsh_accuracy(self, data, beam_width):
    #     input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)
    #     (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

    #     result, _ = self.beam_evaluate(data, beam_width=beam_width)
    #     result = result[:,0]
    #     real = target_tokens[:, 1:]

    #     result_shape = tf.shape(result)
    #     if result_shape[1] > self.max_output_len - 1:
    #         result = tf.slice(result, [0, 0], [result_shape[0], self.max_output_len - 1])
    #     elif result_shape[1] < self.max_output_len - 1:
    #         result = tf.pad(result, [[0,0], [0, self.max_output_len - 1 - result_shape[1]]])

    #     acc = utils.masked_accuracy(real, result, [self.output_padding_token, self.output_start_token, self.output_end_token])

    #     return acc

    # def pairwise_accuracy(self, target_sequences, pred_sequences, match_sc=1., mismatch_sc=-1., gap_open_sc=-1., gap_extend_sc=-0.2):
    #     target_seqs = [s.numpy().decode('UTF-8') for s in target_sequences]
    #     pred_seqs = [s.numpy().decode('UTF-8') for s in pred_sequences]
    #     assert(len(target_seqs) == len(pred_seqs))

    #     sc_sum = 0
    #     len_sum = 0
    #     for pred, targ in zip(pred_seqs, target_seqs):
    #         algn = pairwise2.align.globalms(pred, targ, match_sc, mismatch_sc, gap_open_sc, gap_extend_sc)
    #         if len(algn) >= 1:
    #             sc_sum += algn[0].score
    #         len_sum += len(targ)
    #     if len_sum > 0:
    #         return sc_sum / len_sum
    #     return 0


    # def basecall_merge(self, data, beam_width):
    #     input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)
    #     (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

    #     result, _ = self.beam_evaluate(data, beam_width=beam_width)


    ##
    ## GENERAL
    ##

    # def _preprocess(self, input_data, target_tokens):
    #     input_mask = self._prepare_input_mask(input_data)
    #     target_mask = target_tokens != self.output_end_token

    #     return input_data, input_mask, target_tokens, target_mask

    def _prepare_input_mask(self, input_data, input_data_type):
        if input_data_type == 'joint':
            (raw_input, event_input) = input_data
            input_mask_raw = utils.input_mask(raw_input, self.input_padding_value)
            input_mask_event = utils.input_mask(event_input, self.input_padding_value)
            input_mask = tf.concat((input_mask_raw, input_mask_event), axis=-1)
        else:
            input_mask = utils.input_mask(input_data, self.input_padding_value)

        return input_mask

    def _encode_input(self, input_data, training=False):
        if self.input_data_type == 'joint':
            (raw_input, event_input) = input_data

            raw_mask = self._prepare_input_mask(raw_input, 'raw')
            enc_output_raw, _ = self.encoder_raw(raw_input, training=training)

            event_mask = self._prepare_input_mask(event_input, 'event')
            enc_output_event, _ = self.encoder_event(event_input, training=training)

            enc_output = tf.concat((enc_output_raw, enc_output_event), axis=1)
            input_mask = tf.concat((raw_mask, event_mask), axis=-1)

        elif self.input_data_type == 'raw':
            input_mask = self._prepare_input_mask(input_data, self.input_data_type)
            enc_output, _ = self.encoder_raw(input_data, training=training)

        elif self.input_data_type == 'event':
            input_mask = self._prepare_input_mask(input_data, self.input_data_type)
            enc_output, _ = self.encoder_event(input_data, training=training)

        return enc_output, input_mask
