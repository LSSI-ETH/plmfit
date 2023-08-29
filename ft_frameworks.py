import torch.nn as nn
import torch

def feature_extraction(dataset_dict, head, lr , epochs, optimizer = 'Adam', criterion = nn.MSELoss(), lr_scheduler = False ):
    
    ft_model = head
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr , betas=(0.9, 0.95))
    
    return head


def train():
    return 0
