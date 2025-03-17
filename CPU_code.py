import numpy as np

class Seq2SeqEncoder:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder weights
        self.W = np.random.randn(self.hidden_dim, self.input_dim) * 0.01  # Input to hidden weight
        self.U = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01  # Hidden to hidden weight
        self.b = np.zeros(self.hidden_dim)  # Hidden bias

        # Store hidden states
        self.hidden_states = []

    def forward(self, input_sequence):
        h = np.zeros(self.hidden_dim)  # Initialize hidden state

        for t in range(len(input_sequence)):
            x_t = input_sequence[t]  # Current input token (assuming it's an embedding)

            # Compute the hidden state (simplified RNN cell)
            h = np.tanh(np.dot(self.W, x_t) + np.dot(self.U, h) + self.b)
            self.hidden_states.append(h)

        return h  # Return the final hidden state


class Seq2SeqDecoder:
    def __init__(self, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Decoder weights
        self.W_h = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01  # Hidden to hidden weight
        self.U_h = np.random.randn(self.hidden_dim, self.output_dim) * 0.01  # Input to hidden weight
        self.b_h = np.zeros(self.hidden_dim)  # Hidden bias
        self.W_y = np.random.randn(self.output_dim, self.hidden_dim) * 0.01  # Hidden to output weight
        self.b_y = np.zeros(self.output_dim)  # Output bias

        # Store hidden states
        self.hidden_state = np.zeros(self.hidden_dim)

    def forward(self, target_sequence, encoder_hidden_state):
        self.hidden_state = encoder_hidden_state  # Set initial hidden state from encoder

        outputs = []
        for t in range(len(target_sequence)):
            token = target_sequence[t]  # Current token from target sequence

            # Update the hidden state
            self.hidden_state = np.tanh(np.dot(self.W_h, self.hidden_state) + np.dot(self.U_h, token) + self.b_h)

            # Compute output
            output = np.dot(self.W_y, self.hidden_state) + self.b_y
            outputs.append(output)

        return np.array(outputs)  # Return sequence of outputs


class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = 0.01

    def forward(self, input_sequence, target_sequence):
        # Encoder forward pass
        encoder_hidden_state = self.encoder.forward(input_sequence)

        # Decoder forward pass
        outputs = self.decoder.forward(target_sequence, encoder_hidden_state)

        return outputs  # Return decoder outputs

    def backward(self, input_sequence, target_sequence, outputs):
        # Compute gradients for Encoder and Decoder (simplified)
        decoder_grads = {
            'W_h': np.zeros_like(self.decoder.W_h),
            'U_h': np.zeros_like(self.decoder.U_h),
            'b_h': np.zeros_like(self.decoder.b_h),
            'W_y': np.zeros_like(self.decoder.W_y),
            'b_y': np.zeros_like(self.decoder.b_y)
        }

        encoder_grads = {
            'W': np.zeros_like(self.encoder.W),
            'U': np.zeros_like(self.encoder.U),
            'b': np.zeros_like(self.encoder.b)
        }

        # Compute the loss and gradients
        loss = 0
        for t in range(len(target_sequence)):
            # Get the predicted and actual output
            output = outputs[t]
            target = target_sequence[t]

            # Mean Squared Error loss
            loss += np.sum((output - target) ** 2)

            # Backward pass (simplified)
            doutput = 2 * (output - target)  # Gradient of MSE loss with respect to output
            decoder_grads['W_y'] += np.outer(doutput, self.decoder.hidden_state)
            decoder_grads['b_y'] += doutput

            # Gradients for hidden state (decoder)
            dh = np.dot(self.decoder.W_y.T, doutput) * (1 - self.decoder.hidden_state ** 2)  # tanh derivative
            decoder_grads['W_h'] += np.outer(dh, self.decoder.hidden_state)
            decoder_grads['U_h'] += np.outer(dh, target_sequence[t])  # Assume token as input to the decoder
            decoder_grads['b_h'] += dh

        # Update parameters
        for param_name, grad in decoder_grads.items():
            setattr(self.decoder, param_name, getattr(self.decoder, param_name) - self.learning_rate * grad)

        return loss / len(target_sequence)  # Return average loss per sequence

if __name__ == "__main__":
    # Testing
    input_dim = 64  # Size of each token vector
    hidden_dim = 128  # Size of hidden state
    output_dim = 64  # Size of output (same as input for simplicity)

    # Initialize Encoder and Decoder
    encoder = Seq2SeqEncoder(input_dim, hidden_dim)
    decoder = Seq2SeqDecoder(hidden_dim, output_dim)

    # Initialize Seq2Seq model
    seq2seq = Seq2Seq(encoder, decoder)

    # Generate dummy data for training
    input_sequences = [np.random.randn(10, input_dim) for _ in range(100)]  # 100 sequences of length 10
    target_sequences = [np.random.randn(10, output_dim) for _ in range(100)]  # 100 target sequences of length 10

    # Training loop
    for epoch in range(10):  # Let's train for 10 epochs
        total_loss = 0
        for input_sequence, target_sequence in zip(input_sequences, target_sequences):
            # Forward pass
            outputs = seq2seq.forward(input_sequence, target_sequence)
            
            # Backward pass and compute loss
            loss = seq2seq.backward(input_sequence, target_sequence, outputs)
            total_loss += loss

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(input_sequences)}')


