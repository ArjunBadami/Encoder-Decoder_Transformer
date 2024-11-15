# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#PART 1 MODULES
class Head(nn.Module):
    def __init__(self, n_embd, head_size, masked=False):
        super().__init__()
        #self.n_embd = n_embd
        self.head_size = head_size
        self.masked = masked

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)


    def forward(self, x):
        B,T, C = x.shape
        q = self.query(x)  #(B,T,C)
        k = self.key(x) #(B,T,C)
        v = self.value(x) #(B,T,C)

        wei = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        if(self.masked): #IF YOU WANT TO DO MASKED ATTENTION ONLY
            tril = torch.tril(torch.ones(T, T))
            wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        x = torch.matmul(wei, v)
        return x, wei


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, masked=False):
        super().__init__()
        head_size = n_embd // n_head
        #self.n_embd = n_embd
        #self.masked = masked
        self.heads = nn.ModuleList([Head(n_embd, head_size, masked) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        res, weights = [], []
        #GET WEIGHTS TENSOR BACK FOR SANITY CHECK
        for h in self.heads:
            r, w = h(x)
            res.append(r)
            weights.append(w)
        out = torch.cat(res, dim=-1)
        out = self.proj(out)
        return out, weights

class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_embd, n_head, masked=False)
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(4*n_embd, n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        #FOR ENCODER, DO LAYER NORM PRE
        maps, weights = self.self_attn(self.norm1(src))
        src = src + maps
        src = self.dropout1(src)
        src = self.norm2(src)
        src3 = F.relu(self.fc1(src))
        src3 = self.fc2(src3)
        src = src + src3
        src = self.dropout2(src)
        '''
        src = src + src2
        src = self.norm1(src)
        src3 = F.relu(self.fc1(src))
        src3 = self.fc2(src3)
        src = src + src3
        src = self.norm2(src)
        '''
        return src, weights


class Encoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, block_size):
        super().__init__()
        #self.n_embd = n_embd
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_encoder = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, n_head) for _ in range(n_layer)])
        #self.norm = nn.LayerNorm(n_embd)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_encoder(torch.arange(T))
        x = tok_emb + pos_emb
        weights = []
        for layer in self.layers:
            x, w = layer(x)
            weights = weights + w
        return x, weights



class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        #x = F.softmax(x, dim=1)
        x = self.log_softmax(x)
        return x


class EncoderAndClassifier(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, block_size, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = Encoder(n_layer, n_embd, n_head, vocab_size, block_size)
        self.classifier = Classifier(n_input, n_hidden, n_output)


    def forward(self, x):
        x, _ = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


#PART 2 MODULES
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_embd, n_head, masked=True)
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(4*n_embd, n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, src):
        '''
        src = src + self.self_attn(self.norm1(src))
        src = self.norm2(src)
        src3 = F.relu(self.fc1(src))
        src3 = self.fc2(src3)
        src = src + src3
        '''
        #FOR DECODER, DO LAYER NORM POST
        src2, weights = self.self_attn(src)
        src = src + src2
        src = self.norm1(src)
        src = self.dropout1(src)
        src3 = F.relu(self.fc1(src))
        src3 = self.fc2(src3)
        src = src + src3
        src = self.norm2(src)
        src = self.dropout2(src)
        return src, weights


class Decoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, block_size):
        super().__init__()
        #self.n_embd = n_embd
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_encoder = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([DecoderLayer(n_embd, n_head, dropout=0.0) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

        #self.norm = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_encoder(torch.arange(T))
        x = tok_emb + pos_emb
        weights = []
        for layer in self.layers:
            x, w = layer(x)
            weights = weights + w
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return loss, weights


#PART 3 MODULES
class PositionalEncoding(nn.Module):
    def __init__(self, block_size, n_embd):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(block_size, n_embd)
        self.encoding.requires_grad = False  # We don't want to backprop through the positional encodings

        position = torch.arange(0, block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]


class EncoderMod(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, block_size):
        super().__init__()
        #self.n_embd = n_embd
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_encoder = PositionalEncoding(block_size, n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, n_head, dropout=0.0) for _ in range(n_layer)])
        #self.norm = nn.LayerNorm(n_embd)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embedding(idx)
        x = self.pos_encoder(x)
        #x = tok_emb + pos_emb
        weights = []
        for layer in self.layers:
            x, w = layer(x)
            weights = weights + w
        return x, weights

class EncoderAndClassifierMod(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, block_size, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = EncoderMod(n_layer, n_embd, n_head, vocab_size, block_size)
        self.classifier = Classifier(n_input, n_hidden, n_output, dropout=0.1)


    def forward(self, x):
        x, _ = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x
