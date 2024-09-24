## Training configuration files documentation

When creating training configuration files for PLMFit, ensure that each file includes the `architecture_parameters` section and the `training_parameters` section. Below are the guidelines for setting up the architecture section correctly:

### Architecture parameters

#### Common parameters

*   **network\_type**: Specifies the type of network. It can be either 'linear' or 'mlp'.
    
*   **output\_dim**: Defines the dimension of the output.
    
    *   For regression and binary classification, set this to 1.
        
    *   For multiclass classification, set this to N (the number of classes).
        
*   **task**: Specifies the type of task. It can be either 'regression' or 'classification'.
    

#### Output activation

*   **For binary classification**: Set output\_activation to 'sigmoid'.
        
*   **For multiclass classification**: Set output\_activation to 'softmax'.
        
*   **For regression**: No output\_activation is needed.
        

#### MLP specific parameters

If network\_type is set to 'mlp', you need to provide additional parameters:

*   **hidden\_dim**: Specifies the dimension of the hidden layers.
    
*   **hidden\_activation**: Defines the activation function for the hidden layers. It can be one of the following:
    
    *   'relu'
        
    *   'sigmoid'
        
    *   'softmax'
        
    *   'tanh'
        
*   **hidden\_dropout**: Specifies the dropout rate for the hidden layers.
    

#### Example configuration

Here is an example configuration for a binary classification task using an MLP network:
```
{
  "architecture_parameters": {
    "network_type": "mlp",
    "output_dim": 1,
    "task": "classification",
    "output_activation": "sigmoid",
    "hidden_dim": 128,
    "hidden_activation": "relu",
    "hidden_dropout": 0.5
  }
}
```
And here is an example configuration for a regression task using a linear network:
```
{
  "architecture_parameters": {
    "network_type": "linear",
    "output_dim": 1,
    "task": "regression"
  }
}
```

### Training parameters
When setting up the `training_parameters` section in your configuration files, ensure that each parameter is correctly defined according to the following guidelines:
    
*   **learning_rate**: A number representing the learning rate for the optimizer.
    
*   **epochs**: A number indicating the total number of training epochs.
    
*   **batch_size**: A number specifying the batch size for training.
    
*   **loss_f**: The loss function to use. It can be one of the following:
    
    *   'bce' (Binary Cross-Entropy)
        
    *   'mse' (Mean Squared Error)
        
    *   'cross_entropy' (Cross-Entropy)
        
*   **optimizer**: The optimizer to use. It can be either 'adam' or 'sgd'.

*   **no_classes**: The total number of classes which has to be equal to the output dimension. Only required when performing multiclass classification.
    
*   **val_split**: A float representing the fraction of the data to be used for validation. This is only needed when not defining a split but should always have a value to avoid exceptions.
    
*   **weight_decay**: A number for weight decay or false if no weight decay is desired.
    
*   **warmup_steps**: Currently not supported. Should be left as 0 or skipped.
    
*   **gradient_clipping**: A value between 0 and 1 for gradient clipping or false if not desired.
    
*   **scheduler**: Should be skipped as it is not implemented.
    
*   **scaler**: Must stay as false and has to be included.
    
*   **gradient_accumulation**: An integer for gradient accumulation steps. If not desired, it should be 1 or false.
    
*   **early_stopping**: A number representing the patience (number of epochs) for early stopping or false if not desired.
    
*   **epoch_sizing**: A float between 0.0 and 1.0 indicating the fraction of the total training data to be used in each epoch, or an integer for an absolute number of samples. Note that if you want the model to see 100% of the data, you must set it to 1.0 (including the .0 part); otherwise, it will only see 1 sample.
    
*   **model_output**: Should stay as 'logits', except when doing feature extraction or training one-hot encoding models.
    
#### Example complete configuration
```
{
  "architecture_parameters": {
    "network_type": "linear",
    "output_dim": 1,
    "task": "regression"
  },
  "training_parameters": {
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 4,
    "loss_f": "mse",
    "optimizer": "adam",
    "val_split": 0.2,
    "weight_decay": 0.0001,
    "warmup_steps": 0,
    "gradient_clipping": false,
    "scheduler": false,
    "scaler": false,
    "gradient_accumulation": 1,
    "early_stopping": 5,
    "epoch_sizing": 1.0,
    "model_output": "logits"
  }
}
```