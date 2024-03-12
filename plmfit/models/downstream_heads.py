import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from plmfit.shared_utils.utils import get_activation_function
    
class LinearHead(nn.Module):
    def __init__(self, config):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(config['input_dim'], config['output_dim'])
        # self.dropout = nn.Dropout(config['dropout'])
        self.task = config['task']
        if self.task == 'classification':
            self.activation = get_activation_function(config['activation'])
    
    def forward(self, x):
        x = self.linear(x)
        # x = self.dropout(x)
        if self.task == 'classification':
            x= self.activation(x)
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
        if self.task == 'classification':
            self.layers.append(get_activation_function(config['output_activation']))
        
        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
