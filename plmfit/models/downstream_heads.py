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
    def __init__(self, in_features, hidden_layers ,num_classes,activation_function ,dropout = 0.25):
        super().__init__()
        self.in_ = nn.Linear(in_features, hidden_layers)
        self.out =nn.Linear(hidden_layers, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation_function = None
        if activation_function == 'softmax':
            self.activation_function = nn.Softmax(dim = 1)
        elif activation_function == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        self.init_weights(nn.Module)
        
        
    def init_weights(self, module) -> None:
        init.kaiming_normal_(self.in_.weight)
        self.in_.bias.data.zero_()
        init.kaiming_normal_(self.out.weight)
        self.out.bias.data.zero_()
        
    def forward(self, src ):
        src = src[0]
        src = torch.mean(src , dim = 1)
        src = src/ torch.max(src)
        src = self.in_(src)
        src = F.relu(self.dropout(src))
        return self.out(src)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
