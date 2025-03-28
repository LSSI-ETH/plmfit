import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from plmfit.shared_utils.utils import get_activation_function
from plmfit.shared_utils.random_state import get_random_state
from plmfit.models.custom_transformer import TransformerPWFF

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
        

# This CNN head expects each input sample to be of shape:
# (sequence_length, embedding_dimension)
# It uses a 1D convolution that treats the embedding dimension as channels.
class CNN(nn.Module):
    def __init__(self, config):
        """
        CNN head for sequence inputs.
        Expected config keys:
            - input_dim: embedding dimension (number of channels).
            - num_filters: number of convolutional filters (default: 32)
            - kernel_size: size of the convolution kernel (default: 3)
            - stride: stride for the convolution (default: 1)
            - padding: padding for the convolution (default: 1)
            - dropout: dropout rate after convolution (default: 0.25)
            - output_dim: dimension of the final output.
            - hidden_activation: activation function after convolution.
            - num_conv_layers: number of convolutional layers to stack (default: 1)
            - output_activation (optional): activation function after the final linear layer.
        """
        super(CNN, self).__init__()
        self.task = config['task']
        in_channels = config['input_dim']
        num_filters = config.get('num_filters', 32)
        kernel_size = config.get('kernel_size', 3)
        stride = config.get('stride', 1)
        padding = config.get('padding', 1)
        dropout_rate = config.get('dropout', 0.25)
        num_conv_layers = config.get('num_conv_layers', 1)
        
        # Create a list of convolutional layers.
        self.convs = nn.ModuleList()
        # First conv layer takes input channels from the embedding dimension.
        self.convs.append(nn.Conv1d(in_channels=in_channels, out_channels=num_filters, 
                                    kernel_size=kernel_size, stride=stride, padding=padding))
        # Subsequent layers take num_filters as both input and output channels.
        for _ in range(1, num_conv_layers):
            self.convs.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                                        kernel_size=kernel_size, stride=stride, padding=padding))
        
        # Activation function after each convolution.
        if 'hidden_activation' in config:
            self.activation = get_activation_function(config['hidden_activation'])
        else:
            self.activation = nn.ReLU()
            
        self.dropout = nn.Dropout(dropout_rate)
        # Pool across the sequence dimension after all convolutions.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, config['output_dim'])
        if "output_activation" in config:
            self.output_activation = get_activation_function(config["output_activation"])
        else:
            self.output_activation = None
        self.init_weights()

    def init_weights(self):
        random_state = get_random_state()
        # Initialize each convolutional layer.
        for conv in self.convs:
            init.xavier_uniform_(conv.weight, generator=random_state)
            if conv.bias is not None:
                init.constant_(conv.bias, 0)
        # Initialize the final fully connected layer.
        init.normal_(self.fc.weight, mean=0.0, std=0.01, generator=random_state)
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Expects input x of shape (batch, sequence_length, embedding_dimension).
        Transposes x to (batch, embedding_dimension, sequence_length) before applying Conv1d.
        """
        if torch.backends.mps.is_available():
            x = x.to(torch.float)
        # Transpose to (batch, channels, sequence_length)
        x = x.transpose(1, 2)
        # Apply each convolutional layer with activation and dropout.
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            x = self.dropout(x)
        # Pool the features across the sequence length.
        x = self.pool(x)  # shape: (batch, num_filters, 1)
        x = torch.flatten(x, 1)  # shape: (batch, num_filters)
        x = self.fc(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


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
    elif network_type == "cnn":
        # For CNN head, we assume input_dim corresponds to the number of input channels.
        config["architecture_parameters"]["input_dim"] = input_dim
        model = CNN(config["architecture_parameters"])
    elif network_type == "transformer":
        config["architecture_parameters"]["vocab_size"] = input_dim
        model = TransformerPWFF.from_config(config["architecture_parameters"])
    else:
        raise ValueError("Head type not supported")
    return model
