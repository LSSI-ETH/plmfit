import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from plmfit.shared_utils.utils import get_activation_function
from plmfit.shared_utils.random_state import get_random_state

class LinearHead(nn.Module):
    def __init__(self, config):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(config['input_dim'], config['output_dim'])
        self.task = config['task']
        self.config = config
        # Check if there's an activation function specified for the layer
        if "output_activation" in config:
            random_state = get_random_state()
            # Initialize weights with a normal distribution around zero
            init.normal_(self.linear.weight, mean=0.0, std=0.01, generator=random_state)
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

        # Input Layer
        self.layers.append(nn.Linear(config['input_dim'], config['hidden_dim']))
        self.layers.append(nn.Dropout(config['hidden_dropout']))
        # Check if there's an activation function specified for the layer
        if 'hidden_activation' in config:
            self.layers.append(get_activation_function(config['hidden_activation']))

        # Hidden Layers
        for _ in range(config.get('hidden_layers', 1) - 1):
            self.layers.append(nn.Linear(config['hidden_dim'], config['hidden_dim']))
            self.layers.append(nn.Dropout(config['hidden_dropout']))
            if 'hidden_activation' in config:
                self.layers.append(get_activation_function(config['hidden_activation']))

        # Output Layer
        self.layers.append(nn.Linear(config['hidden_dim'], config['output_dim']))

        # Check if there's an activation function specified for the layer
        if "output_activation" in config:
            self.layers.append(get_activation_function(config['output_activation']))

        self.init_weights()

    def forward(self, x):
        # if device is MPS, convert input to int
        if torch.backends.mps.is_available():
            x = x.to(torch.float)
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        """Initialize weights using Xavier initialization for internal layers 
        and near-zero initialization for the output layer."""
        random_state = get_random_state()
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if i == len(self.layers) - 2:  # Check if it's the output layer
                    # Initialize output layer weights near zero for classification
                    init.normal_(layer.weight, mean=0.0, std=0.01, generator=random_state)
                    init.constant_(layer.bias, 0)
                else:
                    # Xavier initialization for internal layers
                    init.xavier_uniform_(layer.weight, generator=random_state)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.task = config['task']
        self.rnn = nn.RNN(input_size=config['input_dim'], hidden_size=config['hidden_dim'], num_layers=config['num_layers'],
                          batch_first=True, dropout=config['dropout'], bidirectional=config['bidirectional'])
        fc_input_dim = (
            config["hidden_dim"] * 2 if self.rnn.bidirectional else config["hidden_dim"]
        )
        self.fc = nn.Linear(fc_input_dim, config["output_dim"])
        self.activation = None
        if "output_activation" in config:
            self.activation = get_activation_function(config['output_activation'])
        self.init_weights()

    def forward(self, x):
        # if device is MPS, convert input to int
        if torch.backends.mps.is_available():
            x = x.to(torch.float)
        out, _ = self.rnn(x)
        out = self.fc(out)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def init_weights(self):
        """Initialize weights using Xavier initialization for internal layers 
        and near-zero initialization for the output layer."""
        random_state = get_random_state()
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param, generator=random_state)
            else:
                init.constant_(param, 0)
        # Initialize output layer weights
        init.normal_(self.fc.weight, mean=0.0, std=0.01, generator=random_state)
        init.constant_(self.fc.bias, 0)

class AdapterLayer(nn.Module):
    def __init__(self, in_features, bottleneck_dim ,dropout= 0.25 , eps = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(in_features, eps= eps ,elementwise_affine=True)
        self.fc_down = nn.Linear(in_features, bottleneck_dim)
        self.fc_up = nn.Linear(bottleneck_dim, in_features)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        random_state = get_random_state()
        self.ln.weight.data.fill_(0.01)
        init.kaiming_normal_(self.fc_down.weight, generator=random_state)
        self.fc_down.bias.data.zero_()
        init.kaiming_normal_(self.fc_up.weight, generator=random_state)
        self.fc_up.bias.data.zero_()

    def forward(self, src):
        src = self.ln(src)
        src = nn.relu(self.fc_down(src))
        src = self.fc_up(src)
        return self.dropout(src)


def init_head(config, input_dim):
    network_type = config["architecture_parameters"]["network_type"]
    if network_type == "linear":
        config["architecture_parameters"]["input_dim"] = input_dim
        model = LinearHead(config["architecture_parameters"])
    elif network_type == "mlp":
        config["architecture_parameters"]["input_dim"] = input_dim
        model = MLP(config["architecture_parameters"])
    elif network_type == "rnn":
        config["architecture_parameters"]["input_dim"] = input_dim
        model = RNN(config["architecture_parameters"])
    else:
        raise ValueError("Head type not supported")
    return model
