from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score
import torch.nn as nn
import matplotlib.pyplot as plt
from plmfit.models.downstream_heads import LogisticRegression
from plmfit.shared_utils.utils import log_model_info

layer = 'last'
dataset = 'aav'
model = 'progen2-small'
reduction = 'mean'

# Load dataset
dataset_path = f'./plmfit/data/{dataset}/{dataset}_data_full.csv'
output_path = f'./plmfit/data/{dataset}/models'
data = pd.read_csv(dataset_path)

# Load embeddings and scores
file_path = f'./plmfit/data/{dataset}/embeddings/{dataset}_{model}_embs_layer{layer}_{reduction}.pt'
embeddings = torch.load(file_path, map_location=torch.device('cpu'))
embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings
binary_scores = data['binary_score'].values

# Assuming embeddings and binary_scores are already loaded as NumPy arrays
embeddings = torch.tensor(embeddings, dtype=torch.float32)
binary_scores = torch.tensor(binary_scores, dtype=torch.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, binary_scores, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert splits to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
input_dim = X_train.shape[1]
pred_model = LogisticRegression(input_dim, 1)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(pred_model.parameters(),
                            lr=1e-3, weight_decay=1e-5)

Loss = []
Acc = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = pred_model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        # Assuming binary classification with threshold 0.5
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = total_loss / len(train_loader)
    Loss.append(epoch_loss)
    epoch_accuracy = correct_predictions / total_predictions * 100
    Acc.append(epoch_accuracy)

    if (epoch+1) % 10 == 0 or epoch == 0:  # Print for the first epoch and then every 10th epoch
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {epoch_accuracy:.2f}%, Loss: {epoch_loss:.4f}')


# Function to calculate accuracy
def binary_accuracy(y_true, y_pred):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc


# Evaluate the model
pred_model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_list = []
    y_test_list = []
    for inputs, labels in test_loader:
        y_test_list.append(labels)
        y_pred_tag = pred_model(inputs).squeeze()
        y_pred_list.append(y_pred_tag)

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test_list = [a.squeeze().tolist() for a in y_test_list]
    y_pred_list = [item for sublist in y_pred_list for item in sublist]
    y_test_list = [item for sublist in y_test_list for item in sublist]

    # Calculate metrics
    acc = binary_accuracy(torch.tensor(y_test_list), torch.tensor(y_pred_list))
    roc_auc = roc_auc_score(y_test_list, y_pred_list)
    mcc = matthews_corrcoef(y_test_list, np.round(y_pred_list))
    cm = confusion_matrix(y_test_list, np.round(y_pred_list))

# Save the model checkpoint
torch.save(pred_model.state_dict(
), f'{output_path}/logistic_regression_model_{dataset}_{layer}_{reduction}_{model}.pth')

# Data used to log
data_params = {
    "Dataset": dataset,
    "Layer": layer,
    "Reduction method": reduction,
    "Model used": model
}

# Model parameters to log
model_params = {
    "Input Dimension": input_dim,
    "Output Dimension": 1,
    "Optimizer": "SGD",
    "Learning Rate": 1e-3,
    "Weight Decay (L2 Regularization)": 1e-5
}

# Training parameters to log
training_params = {
    "Number of Epochs": num_epochs,
    "Batch Size": 64,  # Assuming this was your batch size
    "Loss Function": "Binary Cross-Entropy"
}

# Evaluation metrics to log
eval_metrics = {
    "Final Accuracy (%)": f"{acc.item()}%",
    "ROC AUC": roc_auc,
    "MCC": mcc,
    "Confusion Matrix": f"\n{cm}"
}

log_file_path = f'{output_path}/logistic_regression_model_{dataset}_{layer}_{reduction}_{model}.txt'

# Log the information
log_model_info(log_file_path, data_params, model_params, training_params, eval_metrics)


plt.plot(Loss)
plt.xlabel("no. of epochs")
plt.ylabel("total loss")
plt.title("Loss")

# Save the figure to a file
plt.savefig(
    f'{output_path}/logistic_regression_model_{dataset}_{layer}_{reduction}_{model}_loss.png', bbox_inches='tight')

plt.close()

plt.plot(Acc)
plt.xlabel("no. of epochs")
plt.ylabel("total accuracy")
plt.title("Accuracy")

# Save the figure to a file
plt.savefig(
    f'{output_path}/logistic_regression_model_{dataset}_{layer}_{reduction}_{model}_accuracy.png', bbox_inches='tight')

plt.close()