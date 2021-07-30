import numpy as np
import tensorflow as tf
from shape_checker import ShapeChecker
from enc_dec_attn import DecoderInput

import utils

class Basecaller:
    def __init__(self, encoder_raw, encoder_event, decoder, input_data_type, input_padding_value, output_text_processor) -> None:
        self.encoder_raw = encoder_raw
        self.encoder_event = encoder_event
        self.decoder = decoder
        self.input_data_type = input_data_type
        self.input_padding_value = input_padding_value
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True
            )
        )
        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token=''
        )
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()
        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string('[START]')
        self.end_token = index_from_string('[END]')
        self.padding_token = index_from_string('')


    def tokens_to_bases_sequence(self, result_tokens):
        shape_checker = ShapeChecker()
        shape_checker(result_tokens, ('batch', 't'))
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        shape_checker(result_text_tokens, ('batch', 't'))

        result_text = tf.strings.reduce_join(result_text_tokens,
                                            axis=1, separator=' ')
        shape_checker(result_text, ('batch'))

        result_text = tf.strings.strip(result_text)
        shape_checker(result_text, ('batch',))
        return result_text

    def sample(self, logits, temperature):
        shape_checker = ShapeChecker()
        # 't' is usually 1 here.
        shape_checker(logits, ('batch', 't', 'vocab'))
        shape_checker(self.token_mask, ('vocab',))

        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits/temperature,
                                                num_samples=1)

        shape_checker(new_tokens, ('batch', 't'))

        return new_tokens

    def basecall_batch_to_tokens(self, input_data, *, max_length=100, temperature=1.0, early_break=True):
        shape_checker = ShapeChecker()

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
        new_tokens = tf.fill([batch_size, 1], self.start_token)
        shape_checker(new_tokens, ('batch', 't1'))

        # Initialize the accumulators
        result_tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
        attention = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        shape_checker(done, ('batch', 't1'))

        # write start tokens at the beginning
        result_tokens = result_tokens.write(0, new_tokens)

        for t in tf.range(1, max_length, 1):
            dec_input = DecoderInput(
                new_tokens=new_tokens,
                enc_output=enc_output,
                mask=input_mask
            )

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
            attention = attention.write(t, dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)
            shape_checker(dec_result.logits, ('batch', 't1', 'vocab'))
            shape_checker(new_tokens, ('batch', 't1'))

            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)

            # Collect the generated tokens
            result_tokens = result_tokens.write(t, new_tokens)

            if early_break and tf.reduce_all(done):
                break

        result_tokens = result_tokens.stack()
        shape_checker(result_tokens, ('t', 'batch', 't0'))
        result_tokens = tf.squeeze(result_tokens, -1)
        result_tokens = tf.transpose(result_tokens, [1, 0])
        shape_checker(result_tokens, ('batch', 't'))

        attention_stack = attention.stack()
        shape_checker(attention_stack, ('t', 'batch', 't1', 's'))

        attention_stack = tf.squeeze(attention_stack, 2)
        shape_checker(attention_stack, ('t', 'batch', 's'))

        attention_stack = tf.transpose(attention_stack, [1, 0, 2])
        shape_checker(attention_stack, ('batch', 't', 's'))

        return {'token_sequences': result_tokens, 'attention': attention_stack}

    @tf.function
    def tf_basecall_batch_to_tokens(self, input_data, max_length=100, early_break=True):
        return self.basecall_batch_to_tokens(input_data, max_length=max_length, early_break=early_break)


    def basecall_batch(self,
                        input_data,
                        *,
                        max_length=100,
                        return_attention=True,
                        temperature=1.0):
        shape_checker = ShapeChecker()

        basecall_tokens_res = self.tf_basecall_batch_to_tokens(input_data, max_length=max_length)
        token_sequences = basecall_tokens_res['token_sequences']
        result_base_sequences = self.tokens_to_bases_sequence(token_sequences)
        shape_checker(result_base_sequences, ('batch',))

        return {'base_sequences': result_base_sequences, 'attention': basecall_tokens_res['attention']}

    @tf.function
    def tf_basecall_batch(self, raw_input, event_input, max_length=100):
        return self.basecall_batch(raw_input, event_input, max_length=max_length)

    def evaluate_batch(self, data):
        input_data, target_sequence = utils.unpack_data_to_input_target(data, self.input_data_type)

        target_token_sequences = self.output_text_processor(target_sequence)

        max_target_length = tf.shape(target_token_sequences)[1]
        translate_res = self.tf_basecall_batch_to_tokens(input_data, max_length=max_target_length, early_break=False)
        pred_token_sequences = translate_res['token_sequences']

        accuracy = utils.masked_accuracy(
            target_token_sequences,
            pred_token_sequences,
            [self.padding_token, self.start_token, self.end_token]
        )
        return accuracy

    def evaluate(self, data):
        accuracies = []
        for raw_batch, events_batch, target_batch in data:
            acc = self.evaluate_batch((raw_batch, events_batch, target_batch))
            accuracies.append(acc.numpy())

        return np.mean(accuracies)
