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
        
class TransformerPWFF(nn.Module):
    """
    A simple classifier architecture that uses a Transformer encoder, then applies
    a position-wise feedforward (pwff) layer over the entire sequence, and finally
    flattens the result for classification.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embd_dim: int,
        nheads: int,
        d_hid: int,
        nlayers: int,
        max_seq_len: int,
        pwff_dim: int,
        dropout: float = 0.5,
        task: str = "classification",
    ):
        super().__init__()

        # Transformer encoder backbone
        self.encoder = TransformerModel(
            vocab_size=vocab_size,
            embd_dim=embd_dim,
            nhead=nheads,
            d_hid=d_hid,
            nlayers=nlayers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.task = task

        self.pwff = nn.Linear(embd_dim, pwff_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.linear = nn.Linear(pwff_dim * max_seq_len, num_classes)

    def from_config(config):
        return TransformerPWFF(
            num_classes=config["output_dim"],
            vocab_size=config["vocab_size"],
            embd_dim=config["embd_dim"],
            nheads=config["nheads"],
            d_hid=config["d_hid"],
            nlayers=config["nlayers"],
            max_seq_len=config["max_seq_len"],
            pwff_dim=config["pwff_dim"],
            dropout=config["dropout"],
            task=config["task"],
        )

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src (Tensor): Shape [batch_size, seq_len].

        Returns:
            Tensor: Logits of shape [batch_size, num_classes].
        """
        # Convert token IDs to int
        src = src.to(torch.int8)

        # Encode the sequence with the Transformer
        enc_output = self.encoder(src)  # [batch_size, seq_len, embd_dim]

        # Apply a feedforward layer to each position
        pwff_output = self.dropout(
            self.act1(self.pwff(enc_output))
        )  # [batch_size, seq_len, pwff_dim]

        # Flatten the sequence dimension
        pwff_flatten = pwff_output.flatten(
            start_dim=1
        )  # [batch_size, seq_len * pwff_dim]

        # Classify
        logits = self.linear(pwff_flatten)  # [batch_size, num_classes]
        return logits