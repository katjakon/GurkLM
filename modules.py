import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=0
            )
    
    def forward(self, x):
        # x is (n_seq, seq_length)
        return self.embed(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, **kwargs) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            **kwargs
        )
    
    def forward(self, query, key, value, key_padding_mask=None):
        return self.mha(query, key, value, key_padding_mask=key_padding_mask)

class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, dim) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=dim), 
             nn.LayerNorm(normalized_shape=dim)]
        ) 
        self.ff = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=dim),
            nn.ReLU()
        )
    
    def forward(self, x, key_padding_mask):
        attn_output, attn_output_weights = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        add = x + attn_output
        norm1 = self.norms[0](add)
        ff_output = self.ff(norm1)
        add = ff_output + norm1
        norm2 = self.norms[1](add)
        return norm2

class ProjectionLayer(nn.Module):

    def __init__(self, dim, vocab_size) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
        # self.softmax =  nn.LogSoftmax(dim=-1)#nn.Softmax(dim=-1)
    
    def forward(self, x):
        output = self.linear(x)
        return output

class FullModel(nn.Module):

    def __init__(self, model_dim, vocab_size, num_heads, n_layers, dropout ):
        super().__init__()
        self.embeddings = Embedding(vocab_size=vocab_size, dim=model_dim)
        self.encoders = nn.ModuleList([
            EncoderBlock(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, dim=model_dim ) 
            for _ in range(n_layers)])
        self.projection = ProjectionLayer(dim=model_dim, vocab_size=vocab_size)
    
    def forward(self, token_ids, key_padding_mask):
        x = self.embeddings(token_ids)
        for enc in self.encoders:
            x = enc(x, key_padding_mask=key_padding_mask)
        output = self.projection(x)
        return output


