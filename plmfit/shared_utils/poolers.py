import torch
from torch import nn


class GeneralPooler(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()

    def forward(self, hidden_states, pooling_method="default"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if pooling_method == "default":
            first_token_tensor = hidden_states[:, 0]
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
        elif pooling_method == "bos":
            pooled_output = hidden_states[:, 0]
        elif pooling_method == "mean":
            pooled_output = torch.mean(hidden_states, dim=1)
        elif pooling_method == "none":
            pooled_output = hidden_states
        return pooled_output
