# 03-Transformer架构源码构建

## 1. Single-Head-Attention

```python
"""
single head attention: scale-dot-product-Attention
"""
from math import sqrt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, *args):

        super(ScaleDotProductAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.qkv = nn.Linear(self.embedding_dim, self.embedding_dim * 3)

    def forward(self, X):  # please insure your input shape like:
        # [batch_size, seq_len, embedding_dim]
        # 1. q,k,v   2.q @ k^t / d_k   3.mask(ignore)   4.softmax scale   5.attention res
        qkv: Tensor = self.qkv(X)
        query, key, value = torch.split(qkv, [self.embedding_dim, self.embedding_dim, self.embedding_dim], dim=-1)
        # print(query.shape)
        # print(key.shape)
        # print(value.shape)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(key.shape[-1])
        scores = F.softmax(attention_scores, dim=-1)

        # if mask is not None:
        #    pass
        # if dropout is not None:
        #    pass

        attention_res = torch.matmul(attention_scores, value)

        return attention_res, attention_scores


if __name__ == '__main__':
    ipt = torch.randn(2, 16, 64)
    model = ScaleDotProductAttention(ipt.shape[-1])
    attention, scores = model(ipt)

    print(model)
    print(attention.shape)
    print(scores.shape)


```



## 2. Multi-Head-self-Attention

```python
"""
multi head attention
"""
from math import sqrt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num, mask=None, dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.head_dim = self.embedding_dim // self.head_num
        self.qkv = nn.Linear(self.embedding_dim, self.embedding_dim * 3)

    def forward(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        qvk = self.qkv(X)
        query, key, value = torch.split(qvk, self.embedding_dim, dim=-1)
        # multi_head_query = torch.split(query, self.head_dim, dim=-1)
        # multi_head_key = torch.split(key, self.head_dim,  dim=-1),
        # multi_head_value = torch.split(value, self.head_dim, dim=-1)
        multi_head_query = query.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        multi_head_key = key.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        multi_head_value = value.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        attention_score = torch.matmul(multi_head_query, multi_head_key.transpose(-2, -1)) / sqrt(self.head_dim)

        scores = torch.softmax(attention_score, dim=-1)
        attention = torch.matmul(scores, multi_head_value)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)


        return attention, scores


if __name__ == '__main__':
    m = MultiHeadAttention(256, 8)
    ipt = torch.randn(2, 16, 256)
    attention, scores = m(ipt)
    print(attention.shape)
    print(scores.shape)


```



## 3. Cross-Multi-Head-Attention



## 4. Masked-Multi-Head-Attention

