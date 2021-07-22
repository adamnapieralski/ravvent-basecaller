import numpy as np
import tensorflow as tf
from shape_checker import ShapeChecker
from enc_dec_attn import DecoderInput, BATCH_SIZE, INPUT_MAX_LEN

class Basecaller:
    def __init__(self, encoder, decoder, output_text_processor) -> None:
        self.encoder = encoder
        self.decoder = decoder
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


    def translate_symbolic(self,
                        raw_input,
                        *,
                        max_length=50,
                        return_attention=True,
                        temperature=1.0):
        shape_checker = ShapeChecker()

        batch_size = tf.shape(raw_input)[0]

        # Encode the input
        shape_checker(raw_input, ('batch', 's'))

        enc_output, enc_state = self.encoder(raw_input)
        shape_checker(enc_output, ('batch', 's', 'enc_units'))
        shape_checker(enc_state, ('batch', 'enc_units'))

        # Initialize the decoder
        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)
        shape_checker(new_tokens, ('batch', 't1'))

        # Initialize the accumulators
        result_tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
        attention = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        shape_checker(done, ('batch', 't1'))

        for t in tf.range(max_length):
            dec_input = DecoderInput(
                new_tokens=new_tokens, enc_output=enc_output, mask=(raw_input != -1))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
            attention = attention.write(t, dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)
            shape_checker(dec_result.logits, ('batch', 't1', 'vocab'))
            shape_checker(new_tokens, ('batch', 't1'))

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens = result_tokens.write(t, new_tokens)

            if tf.reduce_all(done):
                break

        # Convert the list of generated token ids to a list of strings.
        result_tokens = result_tokens.stack()
        shape_checker(result_tokens, ('t', 'batch', 't0'))
        result_tokens = tf.squeeze(result_tokens, -1)
        result_tokens = tf.transpose(result_tokens, [1, 0])
        shape_checker(result_tokens, ('batch', 't'))

        result_base_sequence = self.tokens_to_bases_sequence(result_tokens)
        shape_checker(result_base_sequence, ('batch',))

        if return_attention:
            attention_stack = attention.stack()
            shape_checker(attention_stack, ('t', 'batch', 't1', 's'))

            attention_stack = tf.squeeze(attention_stack, 2)
            shape_checker(attention_stack, ('t', 'batch', 's'))

            attention_stack = tf.transpose(attention_stack, [1, 0, 2])
            shape_checker(attention_stack, ('batch', 't', 's'))

            return {'base_sequence': result_base_sequence, 'attention': attention_stack}
        else:
            return {'base_sequence': result_base_sequence}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.float32, shape=[BATCH_SIZE, INPUT_MAX_LEN]), tf.TensorSpec(dtype=tf.int32, shape=())])
    def tf_translate(self, raw_input, max_length=100):
        return self.translate_symbolic(raw_input, max_length=max_length)
