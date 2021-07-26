import tensorflow as tf
from tensorflow.python.ops.random_ops import categorical

from shape_checker import ShapeChecker

import typing
from typing import Any, Tuple

import utils

BATCH_SIZE = 64
INPUT_MAX_LEN = 150

RANDOM_SEED = 22

tf.random.set_seed(RANDOM_SEED)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                    # Return the sequence and state
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, sequence, state=None):
        shape_checker = ShapeChecker()
        sequence = tf.reshape(sequence, [sequence.shape[0], sequence.shape[1], 1])
        shape_checker(sequence, ('batch', 's', 1))

        # 2. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(sequence, initial_state=state)
        shape_checker(output, ('batch', 's', 'enc_units'))
        shape_checker(state, ('batch', 'enc_units'))

        # 3. Returns the new sequence and its state.
        return output, state

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

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
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                    embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

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
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
        shape_checker(context_vector, ('batch', 't', 'dec_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return DecoderOutput(logits, attention_weights), state

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        shape_checker = ShapeChecker()
        shape_checker(y_true, ('batch', 't'))
        shape_checker(y_pred, ('batch', 't', 'logits'))

        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)
        shape_checker(loss, ('batch', 't'))

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        shape_checker(mask, ('batch', 't'))
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)

class TrainBasecaller(tf.keras.Model):
    def __init__(self, units, embedding_dim, output_text_processor, use_tf_function=True):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(units)
        decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()

        index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token=''
        )
        self.start_token = index_from_string('[START]')
        self.end_token = index_from_string('[END]')
        self.padding_token = index_from_string('')

    def train_step(self, inputs):
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    def test_step(self, inputs):
        return self._tf_test_step(inputs)

    def _preprocess(self, input_sequence, target_sequence):
        # self.shape_checker(input_sequence, ('batch',))
        self.shape_checker(target_sequence, ('batch',))
        input_mask = input_sequence != -1
        # Convert the text to token IDs
        target_tokens = self.output_text_processor(target_sequence)
        self.shape_checker(input_sequence, ('batch', 's'))
        self.shape_checker(target_tokens, ('batch', 't'))

        # Convert IDs to masks.
        target_mask = target_tokens != 0
        self.shape_checker(target_mask, ('batch', 't'))

        return input_sequence, input_mask, target_tokens, target_mask

    def _train_step(self, inputs):
        input_sequence, target_sequence = inputs

        (input_sequence, input_mask, target_tokens, target_mask) = self._preprocess(input_sequence, target_sequence)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_sequence)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length-1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target the target for the decoder's next prediction.

                input_tokens = target_tokens[:, t:t+1]
                target_pred_tokens = target_tokens[:, t+1:t+2]

                step_loss, dec_state, _ = self._loop_step(input_tokens, target_pred_tokens, input_mask, enc_output, dec_state)
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

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.float32, shape=[BATCH_SIZE, INPUT_MAX_LEN]),
                                    tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)

    def _test_step(self, data):
        input_sequence, target_sequence = data

        (input_sequence, input_mask, target_tokens, target_mask) = self._preprocess(input_sequence, target_sequence)
        max_target_length = tf.shape(target_tokens)[1]

        enc_output, enc_state = self.encoder(input_sequence)

        dec_state = enc_state
        val_loss = tf.constant(0.0)

        input_tokens = target_tokens[:, 0:1] # [START] tokens

        # for accuracy measurement, as in Basecaller
        result_tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
        done = tf.zeros([BATCH_SIZE, 1], dtype=tf.bool)
        result_tokens = result_tokens.write(0, input_tokens)

        for t in tf.range(max_target_length-1):
            # input_tokens = target_tokens[:, t:t+2] # teacher forcing

            target_pred_tokens = target_tokens[:, t+1:t+2]

            step_loss, dec_state, input_tokens = self._loop_step(input_tokens, target_pred_tokens, input_mask, enc_output, dec_state, return_prediction_tokens=True)
            val_loss += step_loss

            # accuracy
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), input_tokens)
            done = done | (new_tokens == self.end_token)
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
            [self.padding_token, self.start_token, self.end_token]
        )

        return {'batch_loss': average_loss, 'accuracy': accuracy}

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.float32, shape=[BATCH_SIZE, INPUT_MAX_LEN]),
                                tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_test_step(self, inputs):
        return self._test_step(inputs)

    def _sample_prediction(self, logits, temperature):
        if temperature == 0.0:
            pred_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            pred_tokens = tf.random.categorical(logits / temperature, num_samples=1)

        return pred_tokens



class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])