import tensorflow as tf


class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=vocab_size,
        )
    
    def call(self, inputs, targets=None):
        # Compute logits
        logits = self.token_embedding_table(inputs)

        if targets is None:
            loss = None
        else:
            # Reshape data to calculate loss
            batch, block, channel = logits.shape
            logits = tf.reshape(logits, (batch*block, channel))
            targets = tf.reshape(targets, batch*block)

            # Compute loss
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_function(targets, logits)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        generation = inputs
        for _ in range(max_new_tokens):
            # Generate predictions
            logits, _ = self(generation)
            # Focus on final time step
            logits = logits[:, -1, :]
            # Get probabilities
            probabilities = tf.nn.softmax(logits, axis=1)
            # Extract top probability
            new = tf.random.categorical(probabilities, num_samples=1, dtype=tf.int32)
            # Append to generated text
            generation = tf.concat([generation, new], axis=1)
        return generation
