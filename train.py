#!/usr/bin/env python3

# This script trains a neural network that solves the following task:
# Given an input sequence XY[0-5]+ where X and Y are two given digits,
# the task is to count the number of occurrences of X and Y in the remaining
# substring and then calculate the difference #X - #Y.
#
# Example:
# Input: 1213211
# Output: 2 (3 - 1)
#
# This task is solved with a multi-head attention network.

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

SEQ_LEN = 5
VOCAB_SIZE = 6
NUM_TRAINING_STEPS = 25000
BATCH_SIZE = 64

# This function generates data samples as described at the beginning of the
# script
def get_data_sample(batch_size=1):
    random_seq = torch.randint(low=0, high=VOCAB_SIZE - 1,
                               size=[batch_size, SEQ_LEN + 2])
    
    ############################################################################
    # TODO: Calculate the ground truth output for the random sequence and store
    # it in 'gts'.
    X = random_seq[:, 0]
    Y = random_seq[:, 1]

    # Remaining substring
    tail = random_seq[:, 2:]

    # Count occurrences
    count_X = (tail == X.unsqueeze(1)).sum(dim=1)
    count_Y = (tail == Y.unsqueeze(1)).sum(dim=1)

    # Ground truth difference #X - #Y
    gts = count_X - count_Y
    ############################################################################
    
    # Ensure that GT is non-negative
    ############################################################################
    # TODO: Why is this needed?
    # The raw result of count_x - count_y can be negative if count_y > count_x. 
    # For cross entropy loss, the target should always be non-negative since 
    # they are used as indices. When we add the SEQ_LEN, we shift the range to 
    # only positive integers, since SEQ_LEN is the max possible difference that 
    # |count_x - count_y| can give.
    ############################################################################
    gts += SEQ_LEN
    return random_seq, gts

# Network definition
class Net(nn.Module):
    def __init__(self, num_encoding_layers=1, num_hidden=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, num_hidden)
        positional_encoding = torch.empty([SEQ_LEN + 2, 1])
        nn.init.normal_(positional_encoding)
        self.positional_encoding = nn.Parameter(positional_encoding,
                                                requires_grad=True)
        q = torch.empty([1, num_hidden])
        nn.init.normal_(q)
        self.q = nn.Parameter(q, requires_grad=True)
        self.encoding_layers = torch.nn.ModuleList([
                                EncodingLayer(num_hidden, num_heads)
                                for _ in range(num_encoding_layers) ])
        self.decoding_layer = MultiHeadAttention(num_hidden, num_heads)
        self.c1 = nn.Conv1d(num_hidden + 1, num_hidden, 1)
        self.fc1 = nn.Linear(num_hidden, 2 * SEQ_LEN + 1)

    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        ########################################################################
        # TODO: In the following lines we add a (trainable) positional encoding
        # to our representation. Why is this needed?
        # Answer: A trainable positional encoding is required to learn the order
        # information. It is important here since the model needs to learn the 
        # order information of first and second character in the sequence. The 
        # first and second character cannot be treated as part of the subsequence.
        # 
        # Can you think of another similar task where the positional encoding
        # would not be necessary? 
        # Answer: A task where this might not be required would be in case of 
        # sentiment analysis. In this case, the order does not matter but the 
        # presence of a word is what matters the most.
        ########################################################################
        positional_encoding = self.positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.repeat([B, 1, 1])
        x = torch.cat([x, positional_encoding], axis=-1)
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = x.transpose(1, 2)
        for encoding_layer in self.encoding_layers:
            x = encoding_layer(x)
        q = self.q.unsqueeze(0).repeat([B, 1, 1])
        x = self.decoding_layer(q, x, x)
        x = x.squeeze(1)
        x = self.fc1(x)
        return x

class EncodingLayer(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()

        self.att = MultiHeadAttention(embed_dim=num_hidden, num_heads=num_heads)
        self.c1 = nn.Conv1d(num_hidden, 2 * num_hidden, 1)
        self.c2 = nn.Conv1d(2 * num_hidden, num_hidden, 1)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])

    def forward(self, x):
        x = self.att(x, x, x)
        x = self.norm1(x)
        x1 = x.transpose(1, 2)
        x1 = self.c1(x1)
        x1 = F.relu(x1)
        x1 = self.c2(x1)
        x1 = F.relu(x1)
        x1 = x1.transpose(1, 2)
        x = x + x1
        x = self.norm2(x)
        return x

# The following two classes implement Attention and Multi-Head Attention from
# the paper "Attention Is All You Need" by Ashish Vaswani et al.

################################################################################
# TODO: Summarize the idea of attention in a few sentences. What are Q, K and V?
# Answer: Attention allows the model to focus on different parts of the inputs. 
# Attention assigns weights based on the values (V). The weights are determined
# using a function between query (Q) and key (K).
# Query (Q) represents what we are actually looking for. Key (K) represents what 
# we currently have and value (V) contains the actual content we need to attend to.
#
# In the following lines, you will implement a naive version of Multi-Head
# Attention. Please do not derive from the given structure. If you have ideas
# about how to optimize the implementation you can however note them in a
# comment or provide an additional implementation.
# Note by Farees: I have implemented a different version of MultiHeadAttention 
# called MultiHeadAttentionMerged that uses a single Scaled-Dot-Product Attention
# instead of attention layers for each heads. You can find it below.
################################################################################

class Attention(nn.Module):
    """
    Class that implements Scaled dot-product attention with Linear Projections for Q, K and V.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.

        # Linear Projections for q, k and v
        self.proj_q = nn.Linear(input_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, embed_dim, bias=False)
        ########################################################################

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: First, calculate a trainable linear projection of q, k and v.
        # Then calculate the scaled dot-product attention as described in
        # Section 3.2.1 of the paper.
        ########################################################################

        # Learnable Linear Projections for q, k and v
        Q = self.proj_q(q)
        K = self.proj_k(k)
        V = self.proj_v(v)

        # Scaled dot-product attention
        dk = Q.size(-1)

        # Scaling with 1/sqrt(dk) after matrix multiplication of Q and K
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        # Computing softmax scores
        weights = F.softmax(scores, dim=-1)

        # Finally, Matrix multiplication on softmax outputs and V
        result = torch.matmul(weights, V)

        return result

class MultiHeadAttentionMerged(nn.Module):
    """
    A seperate implementation of Multi Head Attention where we merge all 
    the heads for each input and use a single attention layer instead of attention layers for each heads. 
    Also achieves 100% accuracy and converges quicker with lower error. Training is also faster with 
    training time 00:04:11. Achieves 100% accuracy at 13400th step. Does not contain Linear Projections as in
    MultiHeadAttention since we already use Linear Projections in Scaled dot-product attention
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.proj_out = nn.Linear(embed_dim, embed_dim, bias=False)

        # Use the single-head Attention for each split head
        self.single_head_att = Attention(self.head_dim, self.head_dim)

    def forward(self, q, k, v):
        B, Length_q, _ = q.size()
        _, Length_k, _ = k.size()

        # Split into heads
        Q, K, V = self.split_into_heads(q, B, Length_q), self.split_into_heads(k, B, Length_k), self.split_into_heads(v, B, Length_k)

        # Merge batch and heads for single-head attention
        Q_flat, K_flat, V_flat = self.merge_batch(Q, B, Length_q), self.merge_batch(K, B, Length_k), self.merge_batch(V, B, Length_k)

        # Apply single-head attention
        out_flat = self.single_head_att(Q_flat, K_flat, V_flat)

        # Restore shape
        out = out_flat.view(B, self.num_heads, Length_q, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Length_q, self.embed_dim)

        # Final linear projection
        result = self.proj_out(out)

        return result
    
    def split_into_heads(self, tensor, B, L):
        return tensor.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    
    def merge_batch(self, tensor, B, L):
        return tensor.contiguous().view(B * self.num_heads, L, self.head_dim)

class MultiHeadAttention(nn.Module):
    """
    Class that implements Multi Head Attention as per the paper "Attention Is All You Need" by Ashish Vaswani et al.
    Uses Linear Projections for Q, K and V. Uses seperate Scaled dot-product attention for each heads. Training time: 00:06:57
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        ########################################################################
        # TODO: Add necessary member variables.
        ########################################################################

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V 
        # (This is not actually required here since we already have Linear Projections
        # in the Scaled dot-product attention. I have added it since the task was
        # to implement the Multi-Head Attention as per the paper). 
        # MultiHeadAttentionMerged implementation doesnot use these Linear Projections.
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Scaled dot-product attention modules. One for each head
        self.attentions = nn.ModuleList([
            Attention(self.head_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Output projection
        self.proj_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k, v):
        # q, k, and v are batch-first

        ########################################################################
        # TODO: Implement multi-head attention as described in Section 3.2.2
        # of the paper.
        ########################################################################

        # Getting Batch size and length
        B, Length_q, _ = q.size()
        _, Length_k, _ = k.size()

        # Project inputs using the learnable projection layers.
        Q, K, V = self.proj_q(q), self.proj_k(k), self.proj_v(v)

        # Split into heads: shape [B, num_heads, L, head_dim]
        Q, K, V = self.split_heads(Q, B, Length_q), self.split_heads(K, B, Length_k), self.split_heads(V, B, Length_k)

        # Applying attentions based on number of heads as per the paper's architecture
        head_outputs = []
        for i in range(self.num_heads):
            Qi = Q[:, i]  # [B, Length_q, head_dim]
            Ki = K[:, i]  # [B, Length_k, head_dim]
            Vi = V[:, i]  # [B, Length_k, head_dim]
            out_i = self.attentions[i](Qi, Ki, Vi)  # applying scaled dot-product attention
            head_outputs.append(out_i)

        # Concatenate along feature dim
        out = torch.cat(head_outputs, dim=-1)  # [B, Length_q, embed_dim]

        # Final linear projection
        result = self.proj_out(out)

        return result
    
    def split_heads(self, tensor, B, L):
        return tensor.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

# Instantiate network, loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# Train the network
for i in range(NUM_TRAINING_STEPS):
    inputs, labels = get_data_sample(BATCH_SIZE)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    accuracy = (torch.argmax(outputs, axis=-1) == labels).float().mean()

    if i % 100 == 0:
        print('[%d/%d] loss: %.3f, accuracy: %.3f' %
              (i , NUM_TRAINING_STEPS - 1, loss.item(), accuracy.item()))
    if i == NUM_TRAINING_STEPS - 1:
        print('Final accuracy: %.3f, expected %.3f' %
              (accuracy.item(), 1.0))
        
