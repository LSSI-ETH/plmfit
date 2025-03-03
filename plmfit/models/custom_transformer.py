import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    Adds fixed sine/cosine positional encodings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Shape [seq_len, batch_size, embedding_dim].
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    A learned positional embedding. It takes as input a positions tensor
    of shape [batch_size, seq_len] (with values 0, 1, ..., seq_len-1) and returns
    the corresponding embeddings.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions (Tensor): Shape [batch_size, seq_len] containing position indices.
        """
        if positions.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {positions.size(1)} exceeds maximum length of {self.max_positions}"
            )
        return super().forward(positions)


class TransformerModel(nn.Module):
    """
    A Transformer-based encoder model that embeds token IDs, adds positional information
    (either via fixed sinusoidal encodings or learned embeddings), and passes them through a TransformerEncoder.
    """
    def __init__(
        self,
        vocab_size: int,
        embd_dim: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        max_seq_len: int,
        dropout: float = 0.1,
        positional_embedding: str = "learned",  # choose "sinusoidal" or "learned"
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        self.positional_embedding_choice = positional_embedding

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=padding_idx)

        # Choose positional encoding type
        if positional_embedding == "learned":
            self.pos_encoder = LearnedPositionalEmbedding(num_embeddings=max_seq_len, embedding_dim=embd_dim, padding_idx=padding_idx)
        else:
            self.pos_encoder = PositionalEncoding(embd_dim, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embd_dim,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            norm_first=True,  # pre-layer normalization if desired
            batch_first=False,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src (Tensor): Token IDs of shape [batch_size, seq_len].
            src_mask (Tensor, optional): Attention mask of shape [seq_len, seq_len].

        Returns:
            Tensor of shape [batch_size, seq_len, embd_dim] representing encoded embeddings.
        """
        src = src.to(torch.long)
        # Embed tokens and scale embeddings
        token_embeddings = self.embedding(src) * math.sqrt(self.embd_dim)  # [batch_size, seq_len, embd_dim]

        if self.positional_embedding_choice == "learned":
            # Generate position indices for each token in the batch
            batch_size, seq_len, _ = token_embeddings.size()
            positions = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0).expand(batch_size, seq_len)
            pos_embeddings = self.pos_encoder(positions)  # [batch_size, seq_len, embd_dim]
            # Add learned positional embeddings to token embeddings
            embeddings = token_embeddings + pos_embeddings
            # Transformer expects [seq_len, batch_size, embd_dim]
            embeddings = embeddings.transpose(0, 1)
            encoded = self.transformer_encoder(embeddings, src_mask)
            return encoded.transpose(0, 1)
        else:
            # For sinusoidal encoding, our module expects [seq_len, batch_size, embd_dim]
            embeddings = token_embeddings.transpose(0, 1)
            embeddings = self.pos_encoder(embeddings)
            encoded = self.transformer_encoder(embeddings, src_mask)
            return encoded.transpose(0, 1)