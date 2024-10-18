import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from plmfit.shared_utils.utils import get_activation_function

class LinearHead(nn.Module):
    def __init__(self, config):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(config['input_dim'], config['output_dim'])
        self.task = config['task']
        self.config = config
        # Check if there's an activation function specified for the layer
        if "output_activation" in config:
            # Initialize weights with a normal distribution around zero
            init.normal_(self.linear.weight, mean=0.0, std=0.01)
            # Initialize biases to zero
            init.zeros_(self.linear.bias)

            self.activation = get_activation_function(config['output_activation'])

    def forward(self, x):
        # if device is MPS, convert input to int
        if torch.backends.mps.is_available():
            x = x.to(torch.float)
        x = self.linear(x)
        if "output_activation" in self.config:
            x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.task = config['task']

        self.layers = nn.ModuleList()

        # Hidden Layer
        self.layers.append(nn.Linear(config['input_dim'], config['hidden_dim']))
        self.layers.append(nn.Dropout(config['hidden_dropout']))
        # Check if there's an activation function specified for the layer
        if 'hidden_activation' in config:
            self.layers.append(get_activation_function(config['hidden_activation']))

        # Output Layer
        self.layers.append(nn.Linear(config['hidden_dim'], config['output_dim']))

        # Check if there's an activation function specified for the layer
        if "output_activation" in config:
            self.layers.append(get_activation_function(config['output_activation']))

        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        """Initialize weights using Xavier initialization for internal layers 
        and near-zero initialization for the output layer."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if i == len(self.layers) - 2:  # Check if it's the output layer
                    # Initialize output layer weights near zero for classification
                    init.normal_(layer.weight, mean=0.0, std=0.01)
                    init.constant_(layer.bias, 0)
                else:
                    # Xavier initialization for internal layers
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.task = config['task']
        self.rnn = nn.RNN(input_size=config['input_dim'], hidden_size=config['hidden_dim'], num_layers=config['num_layers'],
                          batch_first=True, dropout=config['dropout'], bidirectional=config['bidirectional'])
        self.fc = nn.Linear(config['hidden_dim'], config['output_dim'])
        self.activation = get_activation_function(config['output_activation'])
        self.init_weights()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        return out

class AdapterLayer(nn.Module):
    def __init__(self, in_features, bottleneck_dim ,dropout= 0.25 , eps = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(in_features, eps= eps ,elementwise_affine=True)
        self.fc_down = nn.Linear(in_features, bottleneck_dim)
        self.fc_up = nn.Linear(bottleneck_dim, in_features)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        
    def init_weights(self):
        self.ln.weight.data.fill_(0.01)
        init.kaiming_normal_(self.fc_down.weight)
        self.fc_down.bias.data.zero_()
        init.kaiming_normal_(self.fc_up.weight)
        self.fc_up.bias.data.zero_()
        
    def forward(self, src):
        src = self.ln(src)
        src = nn.relu(self.fc_down(src))
        src = self.fc_up(src)
        return self.dropout(src)
