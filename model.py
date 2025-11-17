import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1, freeze_encoder=False):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        if freeze_encoder:
            print("Freezing encoder memory weights (Key & Value)...")
            for param in self.transformer.encoder.parameters():
                param.requires_grad = False
            for param in self.src_embedding.parameters():
                param.requires_grad = False

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, tgt_mask):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt_emb, memory, 
                                          tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_padding_mask, 
                                          memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(output)

def generate_masks(src, tgt, pad_idx, device):
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    tgt_subsequent_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
    return src_padding_mask, tgt_padding_mask, tgt_subsequent_mask