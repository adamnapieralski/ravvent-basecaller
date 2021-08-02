import numpy as np
import tensorflow as tf

from shape_checker import ShapeChecker

import typing
from typing import Any, Tuple

import utils

class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units: int, data_type: str):
        '''
        Initialize encoder
            Parameters:
                enc_units (int): Number of encoder inner units (latent dim)
                data_type (str): Type of encoder regarding input data - allowed values: "raw", "event"
        '''
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.data_type = data_type

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                    # Return the sequence and state
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, sequence, state=None):
        shape_checker = ShapeChecker()
        if self.data_type == 'raw':
            shape_checker(sequence, ('batch', 's', 1))
        elif self.data_type == 'event':
            shape_checker(sequence, ('batch', 's', 5))

        # 2. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(sequence, initial_state=state)
        shape_checker(output, ('batch', 's', 'enc_units'))
        shape_checker(state, ('batch', 'enc_units'))

        # 3. Returns the new sequence and its state.
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units_dec, units_enc):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        # TODO same units, lost details on query (dec rnn output)
        self.W1 = tf.keras.layers.Dense(units_enc, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units_enc, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        # shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        # shape_checker(w2_key, ('batch', 's', 'attn_units'))


        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask
        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores = True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights


class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, dec_units, enc_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = 1 # constant

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size, self.embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units, self.enc_units)

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)

        # For step 5. This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)


    def call(self, inputs: DecoderInput, state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        shape_checker(inputs.mask, ('batch', 's'))
        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        # Step 1. Lookup the embeddings
        vectors = self.embedding(inputs.new_tokens)
        shape_checker(vectors, ('batch', 't', 'embedding_dim'))

        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        shape_checker(state, ('batch', 'dec_units'))

        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        # print(rnn_output, inputs.enc_output, inputs.mask)
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
        # shape_checker(context_vector, ('batch', 't', 'dec_units'))
        # shape_checker(attention_weights, ('batch', 't', 's'))

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        # shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        # shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return DecoderOutput(logits, attention_weights), state


class Basecaller(tf.keras.Model):
    def __init__(self, units: int, output_text_processor, input_data_type: str, input_padding_value, teacher_forcing: bool = True, val_teacher_forcing: bool = False):
        super().__init__()
        # Build the encoder and decoder
        encoder_raw = Encoder(units, 'raw')
        encoder_event = Encoder(units, 'event')
        if input_data_type == 'joint':
            decoder = Decoder(output_text_processor.vocabulary_size(), 2 * units, units)
        else:
            decoder = Decoder(output_text_processor.vocabulary_size(), units, units)

        self.encoder_raw = encoder_raw
        self.encoder_event = encoder_event
        self.decoder = decoder
        self.output_text_processor = output_text_processor
        self.input_data_type = input_data_type
        self.input_padding_value = input_padding_value
        self.teacher_forcing = teacher_forcing
        self.val_teacher_forcing = val_teacher_forcing # on validation dataset
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
        self.output_start_token = index_from_string('[START]')
        self.output_end_token = index_from_string('[END]')
        self.output_padding_token = index_from_string('')

        # The test output should never generate padding, unknown, or start.
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()
        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.test_token_mask = token_mask

    def train_step(self, inputs):
        return self._tf_train_step(inputs)

    def test_step(self, inputs):
        return self._tf_val_step(inputs)

    def _preprocess(self, input_data, target_sequence):
        if self.input_data_type == 'joint':
            (raw_input, event_input) = input_data
            self.shape_checker(raw_input, ('batch', 's', None))
            self.shape_checker(event_input, ('batch', 's', None))
            input_mask_raw = utils.input_mask(raw_input, self.input_padding_value)
            input_mask_event = utils.input_mask(event_input, self.input_padding_value)
            input_mask = tf.concat((input_mask_raw, input_mask_event), axis=-1)
        else:
            self.shape_checker(input_data, ('batch', 's', None))
            input_mask = utils.input_mask(input_data, self.input_padding_value)

        # Convert the text to token IDs
        self.shape_checker(target_sequence, ('batch',))
        target_tokens = self.output_text_processor(target_sequence)
        self.shape_checker(target_tokens, ('batch', 't'))

        # Convert IDs to masks.
        target_mask = target_tokens != self.output_padding_token
        self.shape_checker(target_mask, ('batch', 't'))

        return input_data, input_mask, target_tokens, target_mask

    def _train_step(self, data):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            # Encode the input
            if self.input_data_type == 'joint':
                (raw_input, event_input) = input_data

                enc_output_raw, enc_state_raw = self.encoder_raw(raw_input)
                self.shape_checker(enc_output_raw, ('batch', 's', 'enc_units'))
                self.shape_checker(enc_state_raw, ('batch', 'enc_units'))

                enc_output_event, enc_state_event = self.encoder_event(event_input)

                enc_output = tf.concat((enc_output_raw, enc_output_event), axis=1)
                enc_state = tf.concat((enc_state_raw, enc_state_event), axis=1)

            elif self.input_data_type == 'raw':
                enc_output, enc_state= self.encoder_raw(input_data)
                self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
                self.shape_checker(enc_state, ('batch', 'enc_units'))

            elif self.input_data_type == 'event':
                enc_output, enc_state= self.encoder_event(input_data)
                self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
                self.shape_checker(enc_state, ('batch', 'enc_units'))

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            # start tokens
            input_tokens = target_tokens[:, 0:1]
            pred_tokens = input_tokens # initialization, not used

            for t in tf.range(max_target_length-1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target the target for the decoder's next prediction.

                if self.teacher_forcing:
                    input_tokens = target_tokens[:, t:t+1]
                else:
                    input_tokens = pred_tokens

                target_pred_tokens = target_tokens[:, t+1:t+2]

                step_loss, dec_state, pred_tokens = self._loop_step(input_tokens, target_pred_tokens, input_mask, enc_output, dec_state, return_prediction_tokens=True)
                loss = loss + step_loss

            # Average the loss over all non padding tokens.
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'batch_loss': average_loss}

    def _loop_step(self, input_tokens, target_pred_tokens, input_mask, enc_output, dec_state, return_prediction_tokens=False):
        # Run the decoder one step.
        decoder_input = DecoderInput(new_tokens=input_tokens,
                                    enc_output=enc_output,
                                    mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
        self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
        self.shape_checker(dec_state, ('batch', 'dec_units'))

        # `self.loss` returns the total for non-padded tokens
        y = target_pred_tokens
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        pred_tokens = None

        if return_prediction_tokens:
            pred_tokens = self._sample_prediction(dec_result.logits, 1.0)

        return step_loss, dec_state, pred_tokens

    @tf.function
    def _tf_train_step(self, data):
        return self._train_step(data)

    def _val_step(self, data):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        (input_data, input_mask, target_tokens, target_mask) = self._preprocess(input_data, target_sequence)

        if self.input_data_type == 'joint':
            (raw_input, event_input) = input_data

            enc_output_raw, enc_state_raw = self.encoder_raw(raw_input)
            self.shape_checker(enc_output_raw, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state_raw, ('batch', 'enc_units'))

            enc_output_event, enc_state_event = self.encoder_event(event_input)

            enc_output = tf.concat((enc_output_raw, enc_output_event), axis=1)
            enc_state = tf.concat((enc_state_raw, enc_state_event), axis=1)

            batch_size = tf.shape(raw_input)[0]

        elif self.input_data_type == 'raw':
            enc_output, enc_state= self.encoder_raw(input_data)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))
            batch_size = tf.shape(input_data)[0]

        elif self.input_data_type == 'event':
            enc_output, enc_state= self.encoder_event(input_data)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))
            batch_size = tf.shape(input_data)[0]

        dec_state = enc_state
        val_loss = tf.constant(0.0)

        input_tokens = target_tokens[:, 0:1] # [START] tokens
        pred_tokens = input_tokens

        # for accuracy measurement, as in Basecaller
        result_tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)

        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        result_tokens = result_tokens.write(0, input_tokens)

        max_target_length = tf.shape(target_tokens)[1]
        for t in tf.range(max_target_length-1):
            # input_tokens = target_tokens[:, t:t+2] # teacher forcing

            target_pred_tokens = target_tokens[:, t+1:t+2]

            if self.val_teacher_forcing:
                input_tokens = target_tokens[:, t:t+1]
            else:
                input_tokens = pred_tokens

            target_pred_tokens = target_tokens[:, t+1:t+2]

            step_loss, dec_state, pred_tokens = self._loop_step(input_tokens, target_pred_tokens, input_mask, enc_output, dec_state, return_prediction_tokens=True)
            val_loss += step_loss

            # accuracy
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), pred_tokens)
            done = done | (new_tokens == self.output_end_token)
            result_tokens = result_tokens.write(t+1, new_tokens)

        # Average the loss over all non padding tokens.
        average_loss = val_loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # accuracy
        result_tokens = result_tokens.stack()
        result_tokens = tf.squeeze(result_tokens, -1)
        result_tokens = tf.transpose(result_tokens, [1, 0])

        accuracy = utils.masked_accuracy(
            target_tokens,
            result_tokens,
            [self.output_padding_token, self.output_start_token, self.output_end_token]
        )

        return {'batch_loss': average_loss, 'accuracy': accuracy}

    @tf.function
    def _tf_val_step(self, inputs):
        return self._val_step(inputs)

    def _sample_prediction(self, logits, temperature):
        if temperature == 0.0:
            pred_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            pred_tokens = tf.random.categorical(logits / temperature, num_samples=1)

        return pred_tokens

    ## evaluate - test

    def _sample_prediction_masked(self, logits, temperature):
        # 't' is usually 1 here.
        self.shape_checker(logits, ('batch', 't', 'vocab'))
        self.shape_checker(self.test_token_mask, ('vocab',))

        token_mask = self.test_token_mask[tf.newaxis, tf.newaxis, :]
        self.shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.test_token_mask, -np.inf, logits)
        pred_tokens = self._sample_prediction(logits, temperature)
        self.shape_checker(pred_tokens, ('batch', 't'))
        return pred_tokens

    def basecall_batch_to_tokens(self, input_data, *, output_max_length=100, temperature=1.0, early_break=True):
        # Encode the input
        if self.input_data_type == 'joint':
            # input data as (raw, event) tuple
            (raw_input, event_input) = input_data
            enc_output_raw, enc_state_raw = self.encoder_raw(raw_input)
            enc_output_event, enc_state_event = self.encoder_event(event_input)
            enc_output, enc_state = tf.concat((enc_output_raw, enc_output_event), axis=1), tf.concat((enc_state_raw, enc_state_event), axis=1)

            input_mask_raw = utils.input_mask(raw_input, self.input_padding_value)
            input_mask_event = utils.input_mask(event_input, self.input_padding_value)
            input_mask = tf.concat((input_mask_raw, input_mask_event), axis=-1)

            batch_size = tf.shape(raw_input)[0]
        else:
            batch_size = tf.shape(input_data)[0]
            if self.input_data_type == 'raw':
                enc_output_raw, enc_state_raw = self.encoder_raw(input_data)
                enc_output, enc_state = enc_output_raw, enc_state_raw

                input_mask = utils.input_mask(input_data, self.input_padding_value)
            elif self.input_data_type == 'event':
                enc_output_event, enc_state_event = self.encoder_event(input_data)
                enc_output, enc_state = enc_output_event, enc_state_event

                input_mask = utils.input_mask(input_data, self.input_padding_value)

        # Initialize the decoder
        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.output_start_token)
        self.shape_checker(new_tokens, ('batch', 't1'))

        # Initialize the accumulators
        result_tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
        attention = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        self.shape_checker(done, ('batch', 't1'))

        # write start tokens at the beginning
        result_tokens = result_tokens.write(0, new_tokens)

        for t in tf.range(1, output_max_length, 1):
            dec_input = DecoderInput(
                new_tokens=new_tokens,
                enc_output=enc_output,
                mask=input_mask
            )

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
            attention = attention.write(t, dec_result.attention_weights)

            new_tokens = self._sample_prediction_masked(dec_result.logits, temperature)
            self.shape_checker(dec_result.logits, ('batch', 't1', 'vocab'))
            self.shape_checker(new_tokens, ('batch', 't1'))

            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.output_end_token)

            # Collect the generated tokens
            result_tokens = result_tokens.write(t, new_tokens)

            if early_break and tf.reduce_all(done):
                break

        result_tokens = result_tokens.stack()
        self.shape_checker(result_tokens, ('t', 'batch', 't0'))
        result_tokens = tf.squeeze(result_tokens, -1)
        result_tokens = tf.transpose(result_tokens, [1, 0])
        self.shape_checker(result_tokens, ('batch', 't'))

        attention_stack = attention.stack()
        self.shape_checker(attention_stack, ('t', 'batch', 't1', 's'))

        attention_stack = tf.squeeze(attention_stack, 2)
        self.shape_checker(attention_stack, ('t', 'batch', 's'))

        attention_stack = tf.transpose(attention_stack, [1, 0, 2])
        self.shape_checker(attention_stack, ('batch', 't', 's'))

        return {'token_sequences': result_tokens, 'attention': attention_stack}

    @tf.function
    def tf_basecall_batch_to_tokens(self, input_data, output_max_length=100, early_break=True):
        return self.basecall_batch_to_tokens(input_data, output_max_length=output_max_length, early_break=early_break)

    def tokens_to_bases_sequence(self, result_tokens):
        self.shape_checker(result_tokens, ('batch', 't'))
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        self.shape_checker(result_text_tokens, ('batch', 't'))

        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')
        self.shape_checker(result_text, ('batch'))

        result_text = tf.strings.strip(result_text)
        self.shape_checker(result_text, ('batch',))
        return result_text

    def basecall_batch(self, input_data, *, output_max_length=100):
        basecall_tokens_res = self.tf_basecall_batch_to_tokens(input_data, output_max_length=output_max_length)
        token_sequences = basecall_tokens_res['token_sequences']
        result_base_sequences = self.tokens_to_bases_sequence(token_sequences)
        self.shape_checker(result_base_sequences, ('batch',))

        return {'base_sequences': result_base_sequences, 'attention': basecall_tokens_res['attention']}

    def evaluate_test_batch(self, data):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        target_token_sequences = self.output_text_processor(target_sequence)

        max_target_length = tf.shape(target_token_sequences)[1]
        translate_res = self.tf_basecall_batch_to_tokens(input_data, output_max_length=max_target_length, early_break=False)
        pred_token_sequences = translate_res['token_sequences']

        accuracy = utils.masked_accuracy(
            target_token_sequences,
            pred_token_sequences,
            [self.output_padding_token, self.output_start_token, self.output_end_token]
        )
        return accuracy

    def evaluate_test(self, data):
        accuracies = []
        for raw_batch, events_batch, target_batch in data:
            acc = self.evaluate_test_batch((raw_batch, events_batch, target_batch))
            accuracies.append(acc.numpy())

        return np.mean(accuracies)
