import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

vocab = n_vocab + n_special + n_ctx

def gelu(x):
    return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*torch.pow(x, 3))))

def swish(x):
    return x*torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.relu,
    'swish': swish,
    'gelu': gelu
}

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, n_state, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # One difference with the TF version here: we add epsilon outside of sqrt
        return self.g * (x - mean) / (std + self.eps) + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        if rf == 1: #faster 1x1 conv
            self.w = Parameter(torch.ones(nx, nf)) # TODO change to random normal
            self.b = Parameter(torch.zeros(nf))
        else: #was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + [nf]
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_state, n_head, attn_pdrop, resid_pdrop, scale=False):
        super(Attention, self).__init__()
        self.c_attn = Conv1D(n_state*3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.scale = scale
        self.n_head = n_head
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    @staticmethod
    def mask_attn_weights(w):
        n = w.size(-1)
        b = torch.tril(np.ones(n, n)).view(1, 1, n, n)
        return w * b + -1e9*(1-b)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = self.mask_attn_weights(w)
        w = nn.Softmax()(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        new_x_shape = x.size()[:-2] + [np.prod(x.size()[-2:])]
        x = x.view(*new_x_shape) # in Tensorflow version: merge_states
        return x.permute(0, 2, 1, 3)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + [self.n_head, x.size(-1)//self.n_head]
        x = x.view(*new_x_shape) # in Tensorflow version: split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(3, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, nx, n_state, afn, resid_pdrop):
        super(MLP, self).__init__()
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, nx)
        self.act = ACT_FNS[afn]
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


class Block(nn.Module):
    def __init__(self, nx, n_head, attn_pdrop, resid_pdrop, afn, scale=False):
        super(Block, self).__init__()
        self.attn = Attention(nx, nx, n_head, attn_pdrop, resid_pdrop, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(nx, nx*4, afn, resid_pdrop)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        h = self.attn(x)
        h = self.ln_1(x)
        h = self.mlp(x)
        h = self.ln_2(x)
        return h


class Model(nn.Module):
    """ Transformer model """
    def __init__(self, vocab, n_embd, pdrop, n_layers,
                nx, n_head, attn_pdrop, resid_pdrop, afn):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab, n_embd)
        self.drop = nn.Dropout(pdrop)
        self.blocks = clones(Block(nx, n_head, attn_pdrop,
                                   resid_pdrop, afn, scale=True), n_layers)
        self.decoder = nn.Linear(nhid, vocab, bias=False)
        self.decoder.weight = self.embed.weight

    def forward(self, x, m):
        x = x.view(-1, x.size(2), x.size(3))
        m = m.view(-1, m.size(2))
        e = self.embed(x)
        h = e.sum(dim=2)
        for block in self.blocks:
            h = block(h)
        return h