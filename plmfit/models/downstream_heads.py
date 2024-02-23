import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

  
class LinearRegression(nn.Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(config['input_len'], config['output_len'])
        self.dropout = nn.Dropout(config.get('dropout_rate', 0))  # Apply dropout if specified, else default to 0 (no dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

class CnnReg(nn.Module):
    def __init__(self, in_features ,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1 ,1, kernel_size=(3,3), stride=2, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)   
        self.flat = nn.Flatten()
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.fc3 = nn.Linear(144, 70)
        self.fc4 = nn.Linear(70, num_classes)
        self.init_weights(nn.Module)

    def init_weights(self, module) -> None:
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)
        init.kaiming_normal_(self.fc3.weight)
        self.fc3.bias.data.zero_()
        init.kaiming_normal_(self.fc4.weight)
        self.fc4.bias.data.zero_()
        
    def forward(self, src):
        x = self.act1(self.conv1(src))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.act1(self.fc3(x))
        x = self.drop1(x) 
        return self.fc4(x)

    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Dynamically create layers based on the configuration
        for layer_config in config['layers']:
            layer_type = layer_config['type']
            
            if layer_type == 'linear':
                layer = nn.Linear(layer_config['input_len'], layer_config['output_len'])
                self.layers.append(layer)
                
                # Check if there's an activation function specified for the layer
                if 'activation_function' in layer_config:
                    activation_function = self.get_activation_function(layer_config['activation_function'])
                    if activation_function:
                        self.layers.append(activation_function)
            
            elif layer_type == 'dropout':
                layer = nn.Dropout(layer_config['rate'])
                self.layers.append(layer)
                
            else:
                print(f"Unsupported layer type: {layer_type}")
                continue
        
        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_activation_function(self, name):
        """Returns the activation function based on its name."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        # Add more activation functions as needed
        else:
            print(f"Unsupported activation function: {name}")
            return None
        
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
class LogisticRegression(nn.Module):
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        # Validate the config to ensure it has the necessary keys
        required_keys = ['input_len', 'output_len']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing '{key}' in configuration")

        # Extract parameters from the config
        n_inputs = config['input_len']
        n_outputs = config['output_len']

        # Initialize the linear layer with parameters from the config
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
