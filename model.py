# model.py
import torch
import torch.nn as nn
from config import Config

class MedicalBERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        hidden = Config.embedding_dim

        # --- Embeddings ---
        self.token_embed = nn.Embedding(Config.vocab_size, hidden)
        self.pos_embed = nn.Embedding(Config.max_len, hidden)
        self.seg_embed = nn.Embedding(2, hidden)  # BERT-style segment embeddings

        # --- Dropout ---
        self.dropout = nn.Dropout(Config.dropout)

        # --- Transformer (Pre-LayerNorm for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=Config.n_heads,
            dim_feedforward=Config.ff_dim,
            dropout=Config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-Norm: more stable when training from scratch
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.n_layers,
            norm=nn.LayerNorm(hidden)
        )

        # --- Classification head (BERT-style + extra layers) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden),

            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden // 2),

            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Token embeddings
        tok_emb = self.token_embed(x)

        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)

        # Segment embeddings (all zeros → segment 0)
        seg_ids = torch.zeros_like(x)
        seg_emb = self.seg_embed(seg_ids)

        # Combine embeddings
        h = tok_emb + pos_emb + seg_emb
        h = self.dropout(h)

        # Encoder
        enc = self.encoder(h)

        # CLS pooling
        cls_rep = enc[:, 0]

        # Classification
        out = self.classifier(cls_rep)
        return out
