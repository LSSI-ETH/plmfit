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
import json
import torchmetrics.functional as functional
import itertools
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.nn.functional import sigmoid
from sklearn import metrics
import os

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
        file_path = f'./plmfit/data/{data_type}/embeddings/{data_type}_{model}_embs_{layer}_{reduction}/{data_type}_{model}_embs_{layer}_{reduction}.pt'
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

def create_recall_plot(train_recalls,val_recalls):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_recalls, label='Training Recall')
    plt.plot(val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recalls')
    plt.legend()
    return fig
    

def create_loss_plot(training_losses=None, validation_losses=None, json_path=None):
    if json_path:
        with open(json_path, 'r') as file:
            json_data = json.load(file)
            training_losses = json_data['epoch_train_loss']
            validation_losses = json_data['epoch_val_loss']
    fig = plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    return fig


def plot_roc_curve(y_test_list=None, y_pred_list=None, json_path=None):
    if json_path:
        with open(json_path, 'r') as file:
            json_data = json.load(file)
            fpr = json_data['roc_auc_data']['fpr']
            tpr = json_data['roc_auc_data']['tpr']
            roc_auc_val = json_data['roc_auc_data']['roc_auc_val']
    else:
        fpr, tpr, thresholds = roc_curve(y_test_list, y_pred_list)
        roc_auc_val = auc(fpr, tpr)
        tpr = tpr.tolist()
        fpr = fpr.tolist()

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
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc_val": roc_auc_val
    }
    return fig, roc_auc_data


def plot_actual_vs_predicted(y_test_list=None, y_pred_list=None, axis_range=[0, 1], eval_metrics=None, json_path=None):
    if json_path:
        with open(json_path, 'r') as file:
            json_data = json.load(file)
            y_test_list = json_data['pred_data']['actual']
            y_pred_list = json_data['pred_data']['preds']
            eval_metrics = json_data['main']
    fig, ax = plt.subplots(figsize=(8, 8))
    y_test_list = np.asarray(y_test_list, dtype=np.float32).flatten()
    y_pred_list = np.asarray(y_pred_list, dtype=np.float32).flatten()
    ax.scatter(y_test_list, y_pred_list, color='darkorange', alpha=0.1, label='Predicted vs Actual')
    
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

def plot_confusion_matrix_heatmap(cm=None, json_path=None):
    """
    Plots a confusion matrix heatmap.
    """
    if json_path:
        with open(json_path, 'r') as file:
            json_data = json.load(file)
            cm = json_data['main']['confusion_matrix']
            cm = np.array(cm)
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


def evaluate_classification(model, dataloaders_dict, device, model_output='default'):
    model.eval()  # Set the model to evaluation mode
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for inputs, labels, _ in dataloaders_dict['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            if model_output == 'default':
                outputs = model(inputs).squeeze()
            elif model_output == 'logits':
                outputs = model(inputs).logits.squeeze()
            else:
                raise f'Model output "{model_output}" not defined'
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            if labels.dim() > 1:
                labels = labels.squeeze()
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

def evaluate_regression(model, dataloaders_dict, device, model_output='default'):
    model.eval()  # Set the model to evaluation mode
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for inputs, labels, _ in dataloaders_dict['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            if model_output == 'default':
                outputs = model(inputs).squeeze()
            elif model_output == 'logits':
                outputs = model(inputs).logits.squeeze()
            else:
                raise f'Model output "{model_output}" not defined'
            
            # Ensure outputs and labels are at least 1D
            outputs = outputs.squeeze()
            labels = labels.squeeze()

            # Use np.atleast_1d to ensure the arrays are at least 1D
            y_pred_list.append(np.atleast_1d(outputs.detach().cpu().numpy()))
            y_test_list.append(np.atleast_1d(labels.detach().cpu().numpy()))

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

def plot_multilabel_ROC(fpr, tpr, roc_auc, colors = None, labels = None):
    
    # Plot ROC curve
    if colors == None:
        colors = ["darkorange", "green", "red"]
    
    if labels == None:
        labels = ["Mouse", "Cattle", "Bat"]

    fig = plt.figure(figsize=(12, 9))  # Adjust figure size for better visibility
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{labels[i]} (AUC = %0.2f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Adjusted color and style for unity line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)  # Adjust label font size
    plt.ylabel('True Positive Rate', fontsize=14)  # Adjust label font size
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)  # Adjust title font size
    plt.legend(loc="lower right", fontsize=12)  # Adjust legend font size
    plt.grid(True, linestyle='--', alpha=0.7)  # Adjusted grid style for better visibility
    plt.tight_layout()  # Adjust layout for better spacing

    return fig

def plot_multilabel_prec_recall(prec, rec, colors = None, labels = None ):
    
    # Plot ROC curve
    if colors == None:
        colors = ["darkorange", "green", "red"]
    
    if labels == None:
        labels = ["Mouse", "Cattle", "Bat"]

    fig = plt.figure(figsize=(12, 9))  # Adjust figure size for better visibility
    for i in range(len(prec)):
        plt.plot(rec[i], prec[i], color=colors[i], lw=2, label=f'{labels[i]}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)  # Adjust label font size
    plt.ylabel('Precision', fontsize=14)  # Adjust label font size
    plt.title('Precision Recall Curve', fontsize=16)  # Adjust title font size
    plt.legend(loc="lower right", fontsize=12)  # Adjust legend font size
    plt.grid(True, linestyle='--', alpha=0.7)  # Adjusted grid style for better visibility
    plt.tight_layout()  # Adjust layout for better spacing

    return fig

def plot_confusion_matrices(cms, class_names, normalize = False, titles = None, cmap=plt.cm.Blues):
    if titles == None:
        titles = ['Confusion Matrix 1', 'Confusion Matrix 2', 'Confusion Matrix 3']
        
    num_plots = len(cms)
    fig, axes = plt.subplots(1, num_plots, figsize=(16, 6))

    for idx, (cm, title) in enumerate(zip(cms, titles)):
        ax = axes[idx]
        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, fontsize=12)
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=12)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color=color, fontsize=12)
        ax.set_ylabel('True label', fontsize=14)
        ax.set_xlabel('Predicted label', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    return fig

def plot_mcc(thresholds, scores, best_tre, best_score):
    fig = plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
    plt.plot(thresholds, scores, lw=2, label= 'MCC')
    plt.plot(best_tre, best_score, 'ro') # 'ro' for red circle
    plt.text(best_tre, best_score, f'Highest: {best_score:.2f}', fontsize=12, ha='right', va='bottom')
    y_max = 0.5 + best_score.item()/2
    plt.axvline(x=best_tre, color='gray', linestyle='--', ymax= y_max, linewidth=1, c = 'red')
    plt.text(best_tre, -0.9, f'Best Threshold: {best_tre:.2f}', fontsize=12, ha='right')

    plt.xlim([0.0, 1.0])
    plt.ylim([-1.0, 1.00])

    plt.xlim([0.0, 1.0])
    plt.ylim([-1.0, 1.00])

    plt.xlabel('Threshold', fontsize=14)  # Adjust label font size
    plt.ylabel('MCC Score', fontsize=14)  # Adjust label font size
    plt.title('Threshold Determination', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)  # Adjusted grid style for better visibility
    plt.tight_layout()  # Adjust layout for better spacing
    return fig

def get_threshold_MCC(y_pred, y_test, c_type, ignore):
    # Calculate the best treshold utilizing MCC
    best_tre = None
    best_score = -np.inf
    scores = []
    n_class = len(y_pred[0])
    
    for tre in np.linspace(0.01,1,100):
        score = functional.matthews_corrcoef(y_pred,y_test,c_type,threshold = tre,num_labels = n_class, ignore_index = ignore)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_tre = tre

    fig= plot_mcc(np.linspace(0.01,1,100),scores,best_tre,best_score)
    
    return (best_tre,fig)

def plot_exact_accuracy(y_pred,y_test,best_tre,ignore_index = 0.5, logger = None):
    valid_index = ~(y_test == ignore_index)
    y_test = y_test[valid_index]
    # Calculates the number of correct predictions per sequence
    y_pred_tre = sigmoid(torch.clone(y_pred[valid_index]))
    logger.log(y_test)
    logger.log(y_pred_tre)
    y_pred_tre[y_pred_tre >= best_tre] = 1
    y_pred_tre[y_pred_tre < best_tre] = 0

    pred_sum = torch.sum(torch.round(y_pred_tre) == y_test, dim = 1)
    n_correct, frequency = np.unique(pred_sum,return_counts = True)

    # Create bar plot
    fig = plt.figure(figsize=(8, 6))
    bars = plt.bar(n_correct, frequency, color='skyblue')

    # Add labels to the bars
    for bar, label in zip(bars, frequency):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, label,
                ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('# Correct Guesses')
    plt.ylabel('Frequency')
    plt.title('Exact Accuracy')

    # Show plot
    return fig

def plot_mixedlabel_heatmap(y_pred,best_tre,case_mask,title):
        y_pred_tre = torch.floor(y_pred + 1 - best_tre)
        data = y_pred_tre[case_mask]
        
        fig = plt.figure(figsize=(6, 15))
        colors = ['white', 'steelblue']
        cmap = ListedColormap(colors)
        plt.imshow(data, cmap=cmap, interpolation='nearest', aspect='auto')
        plt.xlabel('Species')
        plt.ylabel('Sequences')
        plt.xticks(ticks = [0,1,2,3], labels = ["Mouse Pred","Cattle Pred","Bat Pred","Human Pred"])
        legend = plt.legend(handles=[Patch(facecolor=colors[0], edgecolor='k', label='Non-bind'),
                                    Patch(facecolor=colors[1], edgecolor='k', label='Bind')],
                            loc='upper center', title="", bbox_to_anchor=(0.5, -0.05), handlelength=4, handleheight=4)

        plt.setp(legend.get_title(), fontsize='large')

        # Add vertical lines between different x ticks
        for i in range(data.shape[1] - 1):
            plt.axvline(x=i + 0.5, color='black', linestyle='-', linewidth=1)

        # Add horizontal lines
        for i in range(data.shape[0] - 1):
            plt.axhline(y=i + 0.5, color='black', linestyle='-', linewidth=1)

        # Customize ticks and labels for x-axis on top
        plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False,labelsize="large")
        plt.tight_layout()
        plt.title(title)
        return fig

def mixed_labels_heatmaps(y_pred,y_test,best_tre,ignore_index = 0.5):
    h_maps = {}
    valid_index = ~torch.any(y_test == ignore_index,axis = 1)
    y_test = y_test[valid_index]
    y_pred = y_pred[valid_index]
    mixed_labels, inverse_ind = np.unique(y_test, axis = 0,return_inverse = True)
    
    for i in range(np.max(len(mixed_labels))):
        case = mixed_labels[i]
        if (-1 in case.tolist()) or (case.tolist() == [0,0,0,0]) or (case.tolist() == [1,1,1,1]):
            continue
        case_ind = np.where(inverse_ind == i)
        h_maps['hmaps/'+ str(case)] = plot_mixedlabel_heatmap(y_pred,best_tre,case_ind,str(case))
        
            
    return h_maps

def evaluate_predictions(y_pred,y_test,c_type,n_class = 1, ignore = 0.5, logger = None, only_mixed = False):
    # Initialize dictionaries to save results, and initialize the metrics
    results = {}
    pooled_results = {}
    figures = {}

    metrics = {'Accuracy':functional.accuracy, 'Precision': functional.precision, 'Recall': functional.recall}
    pooled_metrics = {'MCC': functional.matthews_corrcoef, 'Exact Match': functional.exact_match}

    # Calculate the best threshold for classification based on MCC
    best_tre, mcc_fig = get_threshold_MCC(y_pred,y_test,c_type,ignore)
    figures["MCC"] = mcc_fig

    n_class = len(y_pred[0])
    c_type = "multilabel"

    # Calculate scores for all metrics
    for (name,metric) in metrics.items():
        result = metric(y_pred,y_test,c_type, threshold = best_tre, average = "none", num_labels = n_class,ignore_index =ignore)
        results[name] = result.tolist()
        pooled_results[name] = torch.mean(result).item()

    # Calculate binary MCC for all classes
    mcc_list = []
    for i in range(n_class):
        mcc = functional.matthews_corrcoef(y_pred[:,i],y_test[:,i],"binary",threshold = best_tre,ignore_index = ignore )
        mcc_list.append(mcc)
    results["MCC"] = np.array(mcc_list).tolist()

    # Calculate scores for "pooled metrics"
    for (name,metric) in pooled_metrics.items():
        result = metric(y_pred,y_test,c_type, threshold = best_tre, num_labels = n_class,ignore_index = ignore)
        pooled_results[name] = result.item()

    colors = ["darkorange", "green", "red","blue"]
    labels = ["Mouse","Cattle","Bat","Human"]
    # Plot confusion matrix
    cm = functional.confusion_matrix(y_pred,y_test,c_type, threshold = best_tre, num_labels = n_class,ignore_index = ignore)
    cm_fig = plot_confusion_matrices(cm,["Non-bind","Bind"], titles = labels)
    figures["con_mat"] = cm_fig

    # Plot ROC curve
    fpr, tpr, thresholds = functional.roc(y_pred,y_test,c_type, num_labels = n_class,ignore_index = ignore)
    auc = functional.auroc(y_pred,y_test,c_type, num_labels = n_class,average = 'none', ignore_index = ignore)
    roc_fig= plot_multilabel_ROC(fpr, tpr, auc, colors = colors, labels = labels)
    figures["ROC"] = roc_fig

    # Plot Precision Recall curve
    precision, recall, thresholds = functional.precision_recall_curve(y_pred,y_test,c_type, num_labels = n_class,ignore_index = ignore)
    prec_rec_fig= plot_multilabel_prec_recall(precision, recall, colors = colors, labels = labels)
    figures["prec_recall_curve"] = prec_rec_fig

    # Plot exact accuracy
    #exact_acc_fig = plot_exact_accuracy(y_pred,y_test,best_tre, logger = logger)
    #figures["correct_guesses"] = exact_acc_fig

    """
    if not only_mixed:
        # Plot heatmap for mixed labels
        mixed_label_heatmap = mixed_labels_heatmaps(y_pred,y_test,best_tre)
        for (name,hmap) in mixed_label_heatmap.items():
            figures[name] = hmap
    """
    return (results,pooled_results,figures)

def evaluate_multi_label_classification(model, dataloaders_dict, device, logger = None, only_mixed = False):
    # Evaluate the model on the test dataset
    model.eval()
    y_pred = []
    y_test = []
    with torch.no_grad():
        for (embeddings, labels) in dataloaders_dict['test']:
            embeddings = embeddings.to(device)
            labels = labels.to(device).int()
            output = model(embeddings)
            
            y_pred.extend(output.cpu().detach().numpy())
            y_test.extend(labels.cpu().detach().numpy())

    y_pred = torch.tensor(np.array(y_pred))
    y_test = torch.tensor(np.array(y_test)).int()
    n_class = len(y_test[0])
    
    if only_mixed:
        mixed_labels, inverse_ind = np.unique(y_test, axis = 0, return_inverse = True)
        mixed_indices = []

        for i in range(len(mixed_labels)):
            unique_labels = np.unique(mixed_labels[i])
            if len(unique_labels) < (2 + (-1 in unique_labels)):
                continue
                
            case_ind = np.where(inverse_ind == i)
            mixed_indices.extend(case_ind[0].tolist())
            
        y_test = y_test[np.array(mixed_indices)]
        y_pred = y_pred[np.array(mixed_indices)]

    return evaluate_predictions(y_pred,y_test, c_type = "multilabel", n_class = n_class, ignore = -1, logger = logger, only_mixed = only_mixed)

def create_lr_plot(lr_data):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(lr_data)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.title(f'Learning Rate')
    return fig

def hit_rate(y_true, y_pred):
    total_hits = 0
    n_class = len(y_true[0])
    for pred, true in zip(y_pred,y_true):
        count = np.sum(pred == true)
        total_hits += (count == (n_class - np.count_nonzero(true == -1)))

    return total_hits / len(y_pred)

def calculate_average_hit_rate(preds, trues):
    if len(preds) != len(trues) or len(preds[0]) != len(trues[0]):
        raise ValueError("The dimensions of 'preds' and 'trues' matrices must be the same.")

    total_hits = 0
    total_valid_entries = 0
    preds = np.where(preds > 0.5, np.array(1), np.array(0))
    hit_rates = []
    
    for i in range(len(preds)):
        valid_entry = 0
        hit_entry = 0
        for j in range(len(preds[0])):
            if trues[i][j] != -1:
                total_valid_entries += 1
                valid_entry += 1
                if preds[i][j] == trues[i][j]:
                    total_hits += 1
                    hit_entry +=1
        hit_rates.append(hit_entry / valid_entry)


    average_hit_rate = total_hits / total_valid_entries
    return  average_hit_rate , sum(hit_rates) / len(hit_rates) ,  hit_rates.count(1) / len(hit_rates)

def evaluate_predictions(y_true, y_pred):
    mask = y_true != -1
    species = ['Mouse','Cattle','Bat', 'Human']
    species_true_list = []
    species_pred_list = []

    for i, animal in enumerate(species):
        species_true_list.append(y_true[:,i][mask[:,i]])
        species_pred_list.append(y_pred[:,i][mask[:,i]])

    # Initialize dictionaries to save results, and initialize the metrics
    results = {}

    scores = {'Accuracy':metrics.accuracy_score, 'Precision': metrics.precision_score, 
            'Recall': metrics.recall_score, 'MCC': metrics.matthews_corrcoef}

    # Calculate scores for all metrics
    for (name,score) in scores.items():
        results[name] = {}
        for i, animal in enumerate(species):
            result = score(species_true_list[i], species_pred_list[i])
            results[name][animal] = result 
        results[name]['Average'] = np.mean(np.array([value for value in results[name].values()]))

    results['Hit Rate'] = hit_rate(y_true,y_pred)

    #thomas_hit = calculate_average_hit_rate(y_pred,y_true)
    #thomas_hit_names = ['Average Hit Rate', 'Micro Hit Rate', 'Completely Correct']
    #for score, name in zip(thomas_hit, thomas_hit_names):
        #results[name] = score

    return (results)

def plot_dual_bar_chart(data_dict1, data_dict2):
    keys1 = list(data_dict1.keys())
    values1 = list(data_dict1.values())

    keys2 = list(data_dict2.keys())
    values2 = list(data_dict2.values())

    # Generate a list of colors for each set of bars
    num_bars1 = len(keys1)
    colors1 = plt.cm.tab20c(np.linspace(0, 1, num_bars1))

    num_bars2 = len(keys2)
    colors2 = plt.cm.tab20b(np.linspace(0, 1, num_bars2))

    # Increase figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate the position of each set of bars
    bar_width = 1
    x_pos1 = np.arange(len(keys1)) * (bar_width)
    x_pos2 = 5 + (1 + np.arange(len(keys2))) * (bar_width) #* (bar_width) + (len(keys2) + 2) * (bar_width)

    # Plot bars for the first data set
    bars1 = ax.bar(x_pos1, values1, width=bar_width, color=colors1, edgecolor='black', linewidth=1.5, label='Set 1')

    # Plot bars for the second data set
    bars2 = ax.bar(x_pos2, values2, width=bar_width, color=colors2, edgecolor='black', linewidth=1.5, label='Set 2')

    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Performances for All & Mixed Labels', fontsize=14)

    # Set the position of x ticks and labels
    x_pos_combined = np.concatenate((x_pos1, x_pos2))
    ax.set_xticks(x_pos_combined, keys1 + keys2)

    # Add value labels on top of each bar in set 1
    for bar1 in bars1:
        yval1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2, yval1 + 0.05, round(yval1, 2), ha='center', va='bottom', fontsize=10)

    # Add value labels on top of each bar in set 2
    for bar2 in bars2:
        yval2 = bar2.get_height()
        ax.text(bar2.get_x() + bar2.get_width()/2, yval2 + 0.05, round(yval2, 2), ha='center', va='bottom', fontsize=10)

    plt.yticks(fontsize=10)
    
     # Add titles for separated plots
    plt.text(3.0,1.15,'All Ground Truths')
    plt.text(5.5,1.15,'Only Mixed Labels')

    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Set y-axis limits
    ax.set_ylim(0, 1.2)
    
    # A vertical line to divide the plot in half
    ax.axvline(x=5, color='gray', linestyle='--')
    plt.tight_layout()

    return fig

def get_experiment_arguments(experiment_dir):
    # Find the json file that was saved during training
    for file in os.scandir(experiment_dir):
        if file.name.endswith('data.json'):
            with open(file.path, "r") as json_file:
                saved_json = json.load(json_file)
    
    return saved_json

def get_test_loader(experiment_dir, species):
    data = utils.load_dataset("rbd") # Load dataset

    if species != False:
        # Ignore indices that don't have labels for the current species
        data.loc[data[species] == -1,'single_strict_split'] = 'ignore'
    else:
        species = ['mouse','cattle','ihbat','human']

    split = data['single_strict_split'].copy() # Load the splits
    test_dataset = data[data['single_strict_split'] == 'test']
    mixed_mask = (test_dataset['mixed_split'] == 1).values

    

    truths = torch.tensor(data[species].values[split == 'test']).int().cpu()
    args = get_experiment_arguments(experiment_dir)['arguments']
    emb_path = args['output_dir'] + '/extract_embeddings'
    embeddings = utils.load_embeddings(emb_path= emb_path, 
                                       data_type=args['data_type'], 
                                       model=args['plm'], 
                                       layer=args['layer'], 
                                       reduction=args['reduction'])
    
    embs = embeddings[split == 'test'].detach().cpu()

    dataset = torch.utils.data.TensorDataset(embs, truths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    return dataloader, mixed_mask

def save_predictions(predictions, experiment_dir, species):
    names = ['preds','truths','preds_m','truths_m']
    
    if species != False:
        pred_path = f'{experiment_dir}/{species}/predictions'
    else:
        pred_path = f'{experiment_dir}/predictions'

    os.makedirs(pred_path, exist_ok = True)

    for array, name in zip(predictions,names):
        torch.save(torch.from_numpy(array), f'{pred_path}/{name}.pt')

def fill_result_vector(labels, predictions):
    result = np.full_like(labels, -1, dtype = np.float32)
    labels_ind = np.where(labels != -1)[0]
    result[labels_ind] = predictions

    return result

def create_result_vector(experiment_dir, get_mixed = False):
    species = ['mouse','cattle','ihbat','human']
    if get_mixed:
        split_name = 'mixed_split'
        pred_name = 'preds_m.pt'
        split_value = 1
    else:
        split_name = 'single_strict_split'
        pred_name = 'preds.pt'
        split_value = 'test'

    data = utils.load_dataset("rbd")
    split = data[split_name].copy()
    truths = data.loc[split == split_value, species]
    result_vector = np.full_like(truths, -1, dtype = np.float32)

    for i, animal in enumerate(species):
        pred_path = f'{experiment_dir}/{animal}/predictions/{pred_name}'
        pred = torch.load(pred_path, map_location = 'cpu')
        pred = np.array(pred, dtype = np.float32).squeeze()
        result = fill_result_vector(truths[animal].values, pred)
        result_vector[:,i] = result

    return result_vector

def make_predictions(experiment_dir, species, model = None):
    dataloader, mixed_mask = get_test_loader(experiment_dir, species)

    model.cpu()
    model.eval()
    y_pred = []
    y_test = []

    with torch.no_grad():
        for (embeddings, labels) in dataloader:
            output = model(embeddings)

            # Applies sigmoid and round the values for classification
            preds = torch.round(sigmoid(output))
            
            y_pred.extend(preds.cpu().detach().numpy())
            y_test.extend(labels.cpu().detach().numpy())

    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    y_pred_m = y_pred[mixed_mask]
    y_test_m = y_test[mixed_mask]

    predictions = (y_pred, y_test, y_pred_m, y_test_m)
    save_predictions(predictions,experiment_dir,species)

def combine_model_predictions(experiment_dir):
    # At the end of evaluating all the separate models, we want to create a result vector
    y_pred = create_result_vector(experiment_dir, get_mixed = False)
    y_pred_m = create_result_vector(experiment_dir, get_mixed = True)
    dataloader, mixed_mask = get_test_loader(experiment_dir, species = False)
    y_test = np.array(dataloader.dataset.tensors[1])
    y_test_m = y_test[mixed_mask]
    predictions = (y_pred, y_test, y_pred_m, y_test_m)
    save_predictions(predictions,experiment_dir,species = False)

def collect_averages(results):
    averages = {}
    for metric, value in results.items():
        if type(value) == dict and 'Average' in value:
            averages[metric] = value['Average']
        else:
            averages[metric] = value
    return averages

def create_results_json(experiment_dir):
    experiment_dir = 'results/fine_tuning/feature_extraction/rbd_ankh-base_middle_mean/test/'
    result_arrays = []

    for file in os.scandir(f'{experiment_dir}/predictions'):
        tensor = torch.load(file.path)
        result_arrays.append(tensor.numpy())

    (y_pred, y_pred_m, y_test, y_test_m) = result_arrays
    results = evaluate_predictions(y_test, y_pred)
    mixed_results = evaluate_predictions(y_test_m, y_pred_m)

    file_path = f'{experiment_dir}/results.json'
    with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

    file_path = f'{experiment_dir}/mixed_results.json'
    with open(file_path, "w") as json_file:
            json.dump(mixed_results, json_file, indent=4)

    plot_path = f'{experiment_dir}/plots'
    os.makedirs(plot_path, exist_ok= True)
    plot = plot_dual_bar_chart(collect_averages(results),collect_averages(mixed_results))
    plot.savefig(f'{plot_path}/performance.png')