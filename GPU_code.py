import numpy as np
from numba import cuda, float32, float64, void
import math
import numba
from time import time

# ------------------------------
# CUDA Configuration
# ------------------------------
BLOCK_SIZE = 32
STREAM = cuda.stream()

# ------------------------------
# CUDA Kernels

@cuda.jit(device=True)
def fast_tanh(x):
    """Optimized tanh approximation"""
    x = x * 0.5
    x = 2.0 / (1.0 + math.exp(-2.0 * x)) - 1.0
    return x



@cuda.jit
def softmax_kernel(x, output):
    row = cuda.blockIdx.x  # Batch index
    col = cuda.blockIdx.y  # Column index (0 for decoder's 3D reshape)
    tid = cuda.threadIdx.x
    vec_len = x.shape[2]

    # Shared memory for max and sum reductions
    shared_max = cuda.shared.array(shape=1024, dtype=float64)
    shared_sum = cuda.shared.array(shape=1024, dtype=float64)

    # Step 1: Find max value
    max_val = -math.inf
    for i in range(tid, vec_len, cuda.blockDim.x):
        max_val = max(max_val, x[row, col, i])
    shared_max[tid] = max_val
    cuda.syncthreads()

    # Reduce max in shared memory
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s])
        cuda.syncthreads()
        s //= 2
    max_val = shared_max[0]

    # Step 2: Compute sum of exponentials
    sum_exp = 0.0
    for i in range(tid, vec_len, cuda.blockDim.x):
        sum_exp += math.exp(x[row, col, i] - max_val)
    shared_sum[tid] = sum_exp
    cuda.syncthreads()

    # Reduce sum in shared memory
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
        cuda.syncthreads()
        s //= 2
    sum_exp = shared_sum[0]

    # Step 3: Compute softmax values
    for i in range(tid, vec_len, cuda.blockDim.x):
        output[row, col, i] = math.exp(x[row, col, i] - max_val) / sum_exp


@cuda.jit
def fused_rnn_kernel(x, W_ih, h_prev, W_hh, b_ih, b_hh, h_next):
    """Optimized RNN with fused operations"""
    row, col = cuda.grid(2)
    if row < h_next.shape[0] and col < h_next.shape[1]:
        sum_val = 0.0
        
        # Tiled matrix multiplication
        tile_size = 32
        for i in range(0, x.shape[1], tile_size):
            x_tile = x[row, i:i+tile_size]
            w_tile = W_ih[i:i+tile_size, col]
            for j in range(tile_size):
                if i+j < x.shape[1]:
                    sum_val += x_tile[j] * w_tile[j]
        
        for i in range(0, h_prev.shape[1], tile_size):
            h_tile = h_prev[row, i:i+tile_size]
            w_tile = W_hh[i:i+tile_size, col]
            for j in range(tile_size):
                if i+j < h_prev.shape[1]:
                    sum_val += h_tile[j] * w_tile[j]

        sum_val += b_ih[col] + b_hh[col]
        h_next[row, col] = fast_tanh(sum_val)

@cuda.jit
def embedding_kernel(embeddings, indices, output):
    batch_idx = cuda.blockIdx.x
    emb_dim = cuda.threadIdx.x
    
    if batch_idx < output.shape[0] and emb_dim < output.shape[1]:
        output[batch_idx, emb_dim] = embeddings[indices[batch_idx], emb_dim]

@cuda.jit
def linear_layer_kernel(inputs, weights, bias, output):
    row, col = cuda.grid(2)
    if row < output.shape[0] and col < output.shape[1]:
        sum_val = 0.0
        tile_size = 32
        for i in range(0, inputs.shape[1], tile_size):
            in_tile = inputs[row, i:i+tile_size]
            wt_tile = weights[i:i+tile_size, col]
            for j in range(tile_size):
                if i+j < inputs.shape[1]:
                    sum_val += in_tile[j] * wt_tile[j]
        output[row, col] = sum_val + bias[col]

@cuda.jit
def cross_entropy_kernel(probs, targets, loss):
    """Cross-entropy with class indices"""
    t, b = cuda.grid(2)
    if t < probs.shape[0] and b < probs.shape[1]:
        class_idx = targets[t, b]
        prob = probs[t, b, class_idx]
        cuda.atomic.add(loss, 0, -math.log(prob + 1e-8))
@cuda.jit
def copy_kernel(dst, src):
    i, j = cuda.grid(2)
    if i < dst.shape[0] and j < dst.shape[1]:
        dst[i, j] = src[i, j]
@cuda.jit(device=True)
def fast_tanh(x):
    x = x * 0.5
    x = 2.0 / (1.0 + math.exp(-2.0 * x)) - 1.0
    return x

@cuda.jit
def softmax_kernel_3d(x, output):
    row, col = cuda.grid(2)
    if row < x.shape[0] and col < x.shape[1]:
        max_val = -math.inf
        sum_exp = 0.0
        
        # Find max for numerical stability
        for k in range(x.shape[2]):
            val = x[row, col, k]
            if val > max_val:
                max_val = val
                
        # Calculate exponentials
        for k in range(x.shape[2]):
            exp_val = math.exp(x[row, col, k] - max_val)
            sum_exp += exp_val
            
        # Write result
        for k in range(x.shape[2]):
            output[row, col, k] = exp_val / sum_exp

@cuda.jit
def softmax_kernel_2d(x, output):
    row = cuda.grid(1)
    if row < x.shape[0]:
        max_val = -math.inf
        sum_exp = 0.0
        
        # Find max for numerical stability
        for k in range(x.shape[1]):
            val = x[row, k]
            if val > max_val:
                max_val = val
                
        # Calculate exponentials
        for k in range(x.shape[1]):
            exp_val = math.exp(x[row, k] - max_val)
            sum_exp += exp_val
            
        # Write result
        for k in range(x.shape[1]):
            output[row, k] = exp_val / sum_exp

# ------------------------------
# Encoder Component
# ------------------------------
sgd_kernel_signature = void(float64[:], float64[::1]) # lr signature is float64[::1]

@cuda.jit(sgd_kernel_signature)
def sgd_kernel(param, lr):
    i = cuda.grid(1)
    if i < param.shape[0]:
        scalar_lr = lr[0]
        param_i_float64 = float64(param[i]) # Explicit casts - keep for now
        lr_term_float64 = float64(float64(1.0) - scalar_lr * 0.01) # Explicitly cast 1.0 to float64
        param[i] = param_i_float64 * lr_term_float64 # Correct multiplication with update term


class CudaEncoder:
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        # Initialize parameters with proper scaling
        scale = math.sqrt(1. / embedding_dim)
        self.embedding = cuda.to_device(np.random.uniform(
            -scale, scale, (input_dim, embedding_dim)).astype(np.float64))
        
        scale = math.sqrt(2. / (embedding_dim + hidden_dim))
        self.W_ih = cuda.to_device(np.random.uniform(
            -scale, scale, (embedding_dim, hidden_dim)).astype(np.float64))
        
        scale = math.sqrt(1. / hidden_dim)
        self.W_hh = cuda.to_device(np.random.uniform(
            -scale, scale, (hidden_dim, hidden_dim)).astype(np.float64))
        
        self.b_ih = cuda.to_device(np.zeros(hidden_dim, dtype=np.float64))
        self.b_hh = cuda.to_device(np.zeros(hidden_dim, dtype=np.float64))
        
        self.hidden_dim = hidden_dim

    def forward(self, d_input):
        batch_size, seq_len = d_input.shape
        h = cuda.device_array((batch_size, self.hidden_dim), dtype=np.float64, stream=STREAM)
        h_states = cuda.device_array((seq_len, batch_size, self.hidden_dim), dtype=np.float64, stream=STREAM)
        
        embedded = cuda.device_array((batch_size, self.embedding.shape[1]), dtype=np.float64, stream=STREAM)
        
        grid = (batch_size,)
        block = (self.embedding.shape[1],)
        
        for t in range(seq_len):
            embedding_kernel[grid, block, STREAM](
                self.embedding, d_input[:, t], embedded
            )
            
            grid_rnn = (
                (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
                (self.hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
            )
            block_rnn = (BLOCK_SIZE, BLOCK_SIZE)
            
            fused_rnn_kernel[grid_rnn, block_rnn, STREAM](
                embedded, self.W_ih,
                h, self.W_hh,
                self.b_ih, self.b_hh,
                h
            )
            
            # Copy h to h_states[t] using device kernel
            grid_copy = (
                (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
                (self.hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
            )
            copy_kernel[grid_copy, block_rnn, STREAM](h_states[t], h)
        
        return h_states, h

# ------------------------------
# Decoder Component
# ------------------------------

class CudaDecoder:
    def __init__(self, output_dim, embedding_dim, hidden_dim):
        # Xavier initialization
        scale = math.sqrt(2. / (embedding_dim + hidden_dim))
        self.embedding = cuda.to_device(np.random.uniform(
            -scale, scale, (output_dim, embedding_dim)).astype(np.float64))
        
        self.W_ih = cuda.to_device(np.random.uniform(
            -scale, scale, (embedding_dim, hidden_dim)).astype(np.float64))
        
        scale = math.sqrt(2. / (hidden_dim + hidden_dim))
        self.W_hh = cuda.to_device(np.random.uniform(
            -scale, scale, (hidden_dim, hidden_dim)).astype(np.float64))
        
        scale = math.sqrt(1. / hidden_dim)
        self.W_out = cuda.to_device(np.random.uniform(
            -scale, scale, (hidden_dim, output_dim)).astype(np.float64))
        
        self.b_ih = cuda.to_device(np.zeros(hidden_dim, dtype=np.float64))
        self.b_hh = cuda.to_device(np.zeros(hidden_dim, dtype=np.float64))
        self.b_out = cuda.to_device(np.zeros(output_dim, dtype=np.float64))
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, d_input, d_hidden):
        batch_size, seq_len = d_input.shape
        outputs = cuda.device_array((seq_len, batch_size, self.output_dim), dtype=np.float64, stream=STREAM)
        probs = cuda.device_array_like(outputs)
        
        embedded = cuda.device_array((batch_size, self.embedding.shape[1]), dtype=np.float64, stream=STREAM)
        
        grid = (batch_size,)
        block = (self.embedding.shape[1],)
        
        for t in range(seq_len):
            embedding_kernel[grid, block, STREAM](
                self.embedding, d_input[:, t], embedded
            )
            
            grid_rnn = (
                (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
                (self.hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
            )
            block_rnn = (BLOCK_SIZE, BLOCK_SIZE)
            
            fused_rnn_kernel[grid_rnn, block_rnn, STREAM](
                embedded, self.W_ih,
                d_hidden, self.W_hh,
                self.b_ih, self.b_hh,
                d_hidden
            )
            
            grid_out = (
                (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
                (self.output_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
            )
            linear_layer_kernel[grid_out, block_rnn, STREAM](
                d_hidden, self.W_out, self.b_out, outputs[t]
            )
            
            # Apply softmax using 2D kernel
            grid_softmax = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_softmax = BLOCK_SIZE
            softmax_kernel_2d[(grid_softmax,), (block_softmax,), STREAM](
                outputs[t], probs[t]
            )
        
        return probs    
    
# ------------------------------
# Training System
# ------------------------------

class CudaSeq2Seq:
    def __init__(self, src_dim, trg_dim, embed_dim, hidden_dim):
        self.encoder = CudaEncoder(src_dim, embed_dim, hidden_dim)
        self.decoder = CudaDecoder(trg_dim, embed_dim, hidden_dim)

    def forward(self, src, trg):
        _, hidden = self.encoder.forward(src)
        return self.decoder.forward(trg, hidden)
class CudaTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = np.array(lr, dtype=np.float64)
        self._collect_params()

    def _collect_params(self):
        self.params = []
        components = [self.model.encoder, self.model.decoder]
        for comp in components:
            self.params.extend([
                comp.embedding, comp.W_ih, comp.W_hh,
                comp.b_ih, comp.b_hh
            ])
        self.params.extend([
            self.model.decoder.W_out,
            self.model.decoder.b_out
        ])

    def step(self, src, trg_input, trg_output):
        # Transpose targets to [Seq, Batch]
        trg_output_t = cuda.to_device(np.ascontiguousarray(trg_output.copy_to_host().T))

        # Forward pass
        probs = self.model.forward(src, trg_input)

        # Compute loss
        loss = self._compute_loss(probs, trg_output_t)

        # Update parameters
        self._update_params_cpu()

        return loss

    def _compute_loss(self, probs, targets):
        loss_arr = cuda.device_array(1, dtype=np.float64)
        grid = (
            (probs.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (probs.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        cross_entropy_kernel[grid, (BLOCK_SIZE, BLOCK_SIZE), STREAM]( # STREAM is assumed to be defined elsewhere
            probs, targets, loss_arr
        )
        return loss_arr.copy_to_host()[0] / (probs.shape[0] * probs.shape[1])

    def _update_params(self):
        for param in self.params:
            blocks = (param.size + 255) // 256
            sgd_kernel[blocks, 256, STREAM](float64(param), float64(self.lr)) # STREAM is assumed to be defined elsewhere
    def _update_params_cpu(self): # No GPU version for debugging
            for param in self.params:
                # CPU-based parameter update (mimicking sgd_kernel logic)
                lr_np = self.lr  # self.lr is already a NumPy array
                param_np = param.copy_to_host() # Get parameter data to CPU as NumPy array
                for i in range(param_np.shape[0]): # Iterate over 'rows' (or elements if 1D)
                    param_np[i] = param_np[i] * (1.0 - lr_np * 0.01)
                param.copy_to_device(param_np) # Copy updated data back to GPU (if needed later, or remove if purely CPU test)



if __name__ == "__main__":
    # For testing
    SRC_VOCAB = 10000
    TRG_VOCAB = 15000
    EMBED_DIM = 256 
    HIDDEN_DIM = 512
    BATCH_SIZE = 64
    SEQ_LEN = 250

    # Initialize
    model = CudaSeq2Seq(SRC_VOCAB, TRG_VOCAB, EMBED_DIM, HIDDEN_DIM)
    trainer = CudaTrainer(model, lr=0.0001)

    # Sample data
    src = cuda.to_device(np.random.randint(0, SRC_VOCAB, (BATCH_SIZE, SEQ_LEN)))
    trg_in = cuda.to_device(np.random.randint(0, TRG_VOCAB, (BATCH_SIZE, SEQ_LEN)))
    trg_out = cuda.to_device(np.random.randint(0, TRG_VOCAB, (BATCH_SIZE, SEQ_LEN)))

    # Train
    for epoch in range(10):
        start = time()
        loss = trainer.step(src, trg_in, trg_out)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Time: {time()-start:.2f}s")
