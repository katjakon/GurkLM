import math

import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, vocab_size, dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=0
            )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x is (n_seq, seq_length)
        return self.dropout(self.embed(x))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class ResidualConnection(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
    
    def forward(self, output, residual):
        x = output + residual
        return self.norm(x)

class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, dim) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            )
        self.ff = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU()
        )
        self.residuals = nn.ModuleList(
            [ResidualConnection(dim=embed_dim),
             ResidualConnection(dim=embed_dim)]
        ) 
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout),
             nn.Dropout(dropout)]
        )
    
    def forward(self, x, key_padding_mask):
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        dropout1 = self.dropouts[0](attn_output)
        residual1 = self.residuals[0](dropout1, x)
        ff_output = self.ff(residual1)
        dropout2 = self.dropouts[1](ff_output)
        residual2 = self.residuals[1](dropout2, residual1)
        return residual2

class ProjectionLayer(nn.Module):

    def __init__(self, dim, vocab_size) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
        self.norm = nn.LayerNorm(normalized_shape=vocab_size)
    
    def forward(self, x):
        output = self.linear(x)
        return self.norm(output)

class FullModel(nn.Module):

    def __init__(self, model_dim, vocab_size, num_heads, n_layers, dropout, max_len):
        super().__init__()
        self.embeddings = Embedding(vocab_size=vocab_size, dim=model_dim, dropout=dropout)
        self.pe = PositionalEncoding(d_model=model_dim, max_len=max_len, dropout=dropout)
        self.encoders = nn.ModuleList([
            EncoderBlock(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, dim=model_dim ) 
            for _ in range(n_layers)])
        self.projection = ProjectionLayer(dim=model_dim, vocab_size=vocab_size)
    
    def forward(self, token_ids, key_padding_mask, pred_mask=None):
        x = self.embeddings(token_ids)
        x = x + self.pe(x)
        for enc in self.encoders:
            x = enc(x, key_padding_mask=key_padding_mask)
        # Gather inputs which are masked
        output = None
        if pred_mask is not None: # if pred_mask is None, no inputs are masked.
            x_flat = torch.flatten(x, end_dim=1)
            x_flat = x_flat[pred_mask]
            # Predict vocab for masked tokens
            output = self.projection(x_flat)
        return {
            "masked_preds": output,
            "representations": x}


