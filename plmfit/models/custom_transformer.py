import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    """
    A Transformer-based encoder model that embeds token IDs, applies positional encoding,
    and passes them through a TransformerEncoder.

    Args:
        vocab_size (int): Size of the token vocabulary.
        embd_dim (int): Dimensionality of the token embeddings.
        nhead (int): Number of attention heads.
        d_hid (int): Dimensionality of the feedforward network in each Transformer layer.
        nlayers (int): Number of Transformer encoder layers.
        max_seq_len (int): Maximum sequence length (for positional encoding).
        dropout (float): Dropout probability.
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
    ):
        super().__init__()

        self.embd_dim = embd_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embd_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embd_dim, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embd_dim,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            norm_first=True,  # if you want pre-layer normalization
            batch_first=False,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src (Tensor): Token IDs of shape [batch_size, seq_len].
            src_mask (Tensor, optional): Attention mask of shape [seq_len, seq_len].

        Returns:
            Tensor of shape [batch_size, seq_len, embd_dim] representing
            encoded (transformed) embeddings.
        """
        # if device is MPS, convert input to long
        if torch.backends.mps.is_available():
            src = src.to(torch.long)

        # src -> [batch_size, seq_len]
        # Embed and scale by sqrt(embd_dim) for better variance
        src = self.embedding(src) * math.sqrt(
            self.embd_dim
        )  # [batch_size, seq_len, embd_dim]

        # TransformerEncoder with batch_first=False expects [seq_len, batch_size, embd_dim].
        src = src.transpose(0, 1)  # -> [seq_len, batch_size, embd_dim]

        # Positional encoding: also expects [seq_len, batch_size, embd_dim]
        src = self.pos_encoder(src)

        # Pass through TransformerEncoder
        output = self.transformer_encoder(
            src, src_mask
        )  # [seq_len, batch_size, embd_dim]

        # Transpose back to [batch_size, seq_len, embd_dim] for downstream usage
        return output.transpose(0, 1)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to each token embedding via sine/cosine functions.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix 'pe' of shape [max_len, 1, d_model].
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # register_buffer ensures 'pe' is not a learnable parameter but is moved with the model's device
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
    An alternative to PositionalEncoding that learns positional embeddings up to a fixed max size.
    Not used in the current model, but provided for completeness.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (Tensor): Shape [batch_size, seq_len].
        """
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} exceeds maximum "
                f"length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


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
