import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plmfit.shared_utils.utils as utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, matthews_corrcoef
import torch
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy.stats import spearmanr


def plot_label_distribution(data, label="binary_score", path=None, text="Keep"):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=label, data=data, hue=label,
                       palette=["coral", "skyblue"])
    plt.title('Label Distribution', fontsize=16)
    plt.xlabel(text + ' Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)

    # Annotate each bar with the count value
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')
    else:
        plt.ion()
        plt.show()


def plot_score_distribution(data, column="score", text="Fitness Score", log_scale=False, path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=1000, kde=True,
                 log_scale=(False, log_scale))
    plt.title(text + ' Distribution', fontsize=16)
    plt.xlabel(text, fontsize=14)
    y_label = 'Frequency (Log Scale)' if log_scale else 'Frequency'
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')
    else:
        plt.ion()
        plt.show()


def normalized_score(data, column="score"):
    # Calculate the minimum and maximum values of the score column
    min_score = data[column].min()
    max_score = data[column].max()

    # Apply Min-Max Normalization
    return (data[column] - min_score) / (max_score - min_score)


def plot_normalized_score_distribution(data, column="normalized_score", text="Fitness Score", log_scale=False, path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=1000, kde=True,
                 log_scale=(False, log_scale))
    plt.title('Normalized ' + text + ' Distribution', fontsize=16)
    plt.xlabel('Normalized ' + text, fontsize=14)
    # Set the y-axis label based on whether log scale is used
    y_label = 'Frequency (Log Scale)' if log_scale else 'Frequency'
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')
    else:
        plt.ion()
        plt.show()


def plot_sequence_length_distribution(data, path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data['sequence_length'], discrete=True)
    plt.title('Sequence Length Distribution', fontsize=16)
    plt.xlabel('Length (Number of Amino Acids)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')
    else:
        plt.ion()
        plt.show()


def plot_mutations_number(data, column='number_of_mutations', annotation=False, path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data[column], color='mediumpurple')
    plt.title('Distribution of Mutation Count', fontsize=16)
    plt.xlabel('Number of Mutations', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    if annotation:
        for p in ax.patches:
            if int(p.get_height()) > 0:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                            textcoords='offset points')
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')
    else:
        plt.ion()
        plt.show()


def parse_fasta(fasta_file, log=False):
    with open(fasta_file, 'r') as file:
        sequence_id = ''
        sequence = ''
        for line in file:
            if line.startswith('>'):
                # Removes the '>' and any trailing newline characters
                sequence_id = line[1:].strip()
            else:
                sequence += line.strip()  # Adds the sequence line, removing any trailing newlines
        if log:
            print("Sequence ID:", sequence_id)
            print("Sequence:", sequence)
        return sequence_id, sequence


def plot_mutations_heatmap(mutation_counts, zoom_region=None, path=None):
    sns.set(style="white")
    mutation_df = pd.DataFrame(mutation_counts)
    fig = plt.figure()
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(
        20, 10), gridspec_kw={'width_ratios': [3, 1]})
    sns.heatmap(np.transpose(mutation_df),
                cmap='viridis', cbar=True, ax=ax_main)
    ax_main.set_title(
        'Mutation Heatmap per Amino Acid and Position', fontsize=16)
    ax_main.set_xlabel('Position in Sequence', fontsize=14)
    ax_main.set_ylabel('Amino Acids', fontsize=14)

    if zoom_region is not None:
        start, end = zoom_region
        sns.heatmap(np.transpose(
            mutation_df.iloc[start:end, :]), ax=ax_zoom, cmap='viridis', cbar=True)
        ax_zoom.set_title(
            f'Zoomed Region: Positions {start} to {end}', fontsize=16)
        ax_zoom.set_xlabel('Position in Sequence', fontsize=14)
        ax_zoom.set_ylabel('Amino Acids', fontsize=14)
    else:
        ax_zoom.axis('off')

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')
    else:
        plt.ion()
        plt.show()


def PCA_2d(data_type, model, layers, reduction, output_path='default', labels_col='score', labeling='continuous', custom_data=None, scaled=True):
    if output_path == 'default':
        output_path = f'./plmfit/data/{data_type}/embeddings/plots'
    else:
        output_path = output_path

    data = utils.load_dataset(
        data_type) if custom_data is None else custom_data

    if labeling == 'discrete':
        # Mapping species to colors for the filtered dataset
        labels_unique = data[labels_col].unique()
        labels_to_int = {labels: i for i, labels in enumerate(labels_unique)}
        labels_colors = data[labels_col].map(labels_to_int).values

        # Generate a discrete colormap
        num_labels = len(labels_unique)
        c = labels_colors
        cmap = plt.get_cmap('tab20', num_labels)  # Using 'tab20' colormap
    else:
        scores = data[labels_col].values
        if scaled:
            scaler = MinMaxScaler()
            scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        c = scores
        cmap = 'viridis'

    for layer in layers:
        # Load embeddings
        file_path = f'./plmfit/data/{data_type}/embeddings/{data_type}_{model}_embs_layer{layer}_{reduction}.pt'
        embeddings = torch.load(file_path, map_location=torch.device('cpu'))
        embeddings = embeddings.numpy() if embeddings.is_cuda else embeddings

        # Perform PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=c, cmap=cmap)
        plt.title(
            f"2D PCA of {data_type} Embeddings\n{labels_col} coloring - Layer {layer} - {reduction} - {model}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        if labeling == 'continuous':
            plt.colorbar(scatter, label=f'{labels_col}')
        else:
            # Create a color bar with tick marks and labels for each species
            cbar = plt.colorbar(scatter, ticks=range(num_labels))
            cbar.set_ticklabels(labels_unique)
            cbar.set_label(f'{labels_col}')

        # Save the figure to a file
        plt.savefig(
            f'{output_path}/PCA_{data_type}_{model}_Layer-{layer}_{reduction}.png', bbox_inches='tight')

        plt.close()


def create_loss_plot(training_losses, validation_losses):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    return fig


def plot_roc_curve(y_test_list, y_pred_list):
    fpr, tpr, thresholds = roc_curve(y_test_list, y_pred_list)
    roc_auc_val = auc(fpr, tpr)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    roc_auc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc_val": roc_auc_val
    }
    return fig, roc_auc_data


def plot_actual_vs_predicted(y_test_list, y_pred_list, axis_range=[0, 1], eval_metrics=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_list, y_pred_list, color='darkorange', edgecolors='k', label='Predicted vs Actual')
    
    min_val = min(min(y_test_list), min(y_pred_list), axis_range[0])
    max_val = max(max(y_test_list), max(y_pred_list), axis_range[1])
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs. Predicted')
    ax.legend(loc='upper left')

    # Display selected metrics in the top right of the graph
    metrics_text = f"R²: {eval_metrics['r_sq']:.3f}\nRMSE: {eval_metrics['rmse']:.3f}\nSpearman's: {eval_metrics['spearman']:.3f}"
    ax.text(0.95, 0.05, metrics_text, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", alpha=0.5, facecolor='white'))

    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(cm):
    """
    Plots a confusion matrix heatmap.
    """
    
    # Convert to percentage
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_names = [0, 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()
    return fig



def binary_accuracy(y_true, y_pred):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc


def evaluate_classification(model, dataloaders_dict, device):
    model.eval()  # Set the model to evaluation mode
    y_pred_list = []
    y_test_list = []
    for inputs, labels in dataloaders_dict['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        y_pred = outputs
        y_pred_list.extend(y_pred.detach().cpu().numpy())
        y_test_list.extend(labels.detach().cpu().numpy())

    # Calculate metrics
    acc = binary_accuracy(torch.tensor(y_test_list), torch.tensor(y_pred_list))
    roc_auc = roc_auc_score(y_test_list, y_pred_list)
    mcc = matthews_corrcoef(y_test_list, np.round(y_pred_list))
    cm = confusion_matrix(y_test_list, np.round(y_pred_list))

    fig, roc_auc_data  = plot_roc_curve(y_test_list, y_pred_list)
    cm_fig = plot_confusion_matrix_heatmap(cm)

    cm = cm.tolist()

    # Assuming cm is your confusion matrix as a numpy array
    cm_dict = {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1])
    }

    # Log evaluation metrics
    eval_metrics = {
        "accuracy": acc.item(),
        "roc_auc": roc_auc,
        "mcc": mcc,
        "confusion_matrix": cm_dict
    }

    return eval_metrics, fig, cm_fig, roc_auc_data

def evaluate_regression(model, dataloaders_dict, device):
    model.eval()  # Set the model to evaluation mode
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for inputs, labels in dataloaders_dict['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            # Directly append numpy arrays to lists
            y_pred_list.append(outputs.detach().cpu().numpy())
            y_test_list.append(labels.detach().cpu().numpy())

    # Convert list of arrays to a single numpy array and then to list for JSON serialization
    y_pred_list = np.concatenate(y_pred_list).tolist()
    y_test_list = np.concatenate(y_test_list).tolist()

    # Calculate metrics
    mse = mean_squared_error(y_test_list, y_pred_list)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_list, y_pred_list)
    r2 = r2_score(y_test_list, y_pred_list)
    spearman_corr = spearmanr(y_test_list, y_pred_list).correlation

    eval_metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r_sq": float(r2),
        "spearman": float(spearman_corr)  # Ensure spearman correlation is a float
    }

    # Assuming plot_actual_vs_predicted is defined elsewhere and returns a matplotlib figure
    fig = plot_actual_vs_predicted(y_test_list, y_pred_list, eval_metrics=eval_metrics)

    testing_data = {
        "y_test": y_test_list,
        "y_pred": y_pred_list,
        "eval_metrics": eval_metrics
    }

    return eval_metrics, fig, testing_data
