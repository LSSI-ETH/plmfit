# PLMFit

PLMFit is a framework that facilitates finetuning of  protein Language Models (currently supporting ProGen) on custom/experimental sequence data with one of the following methods. PLMFit is a powerful tool for working with protein sequences and conducting various tasks, including fine-tuning, task-specific head usage, and feature extraction from supported Protein Language Models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Task-Specific Head Concatenation](#task-specific-head-concatenation)
  - [Fine-Tuning](#fine-tuning)
  - [Feature Extraction](#feature-extraction)

## Installation

To install and use the PLMFit package, follow these steps:

### Prerequisites

Before installing PLMFit, ensure you have Python installed on your system. It's also recommended to use a virtual environment for installation to avoid conflicts with other Python packages.

### Steps

1. **Clone the Repository**

   First, clone the PLMFit repository from GitHub to your local machine:

   ```shell
   git clone https://github.com/LSSI-ETH/plmfit.git

2. **Navigate to the Project Directory**
   ```shell
   cd plmfit

3. **Create and Activate a Virtual Environment (Optional but Recommended)**

   Create a new virtual environment in the project directory:
   ```shell
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows:
   ```shell
   venv\Scripts\activate
   ```

   - On macOS and Linux:
   ```shell
   source venv/bin/activate
   ```

   - On Scrum (Euler):
   ```shell
   source venv/bin/activate
   module load gcc/8.2.0  python_gpu/3.11.2
   ```

5. **Install PLMFit**
   ```shell
   pip install .

## Usage

This section provides an overview of how to use the PLMFit package for various tasks.

**Currently supports**
| PLM          | Versions                                  | Publication Date | Source Link                                           | Owner      |
| ------------ | ----------------------------------------- | ---------------- | ----------------------------------------------------- | ---------- |
| ProGen       | progen2-small, progen2-medium, progen2-xlarge | 2022-01-01       | [Source Link 1](https://github.com/salesforce/progen) | Salesforce |
| ESM          | v2.0, v2.1                                | 2022-02-15       | [Source Link 2](link2)                               | Meta       |
| Ankh         | ankh-base, ankh-large                     | 2023-01-18       |                                                       | Proteinea  |
| Antiberty    | antiberty         | 2021-12-14       | [Source Link 3](https://github.com/jeffreyruffolo/AntiBERTy)                               | Open-source |



### Feature / Embeddings Extraction

The process of leveraging pre-trained language models (PLMs) for specific tasks often involves two key steps: extracting features (or embeddings) from the PLM and then training a "head" (a new model layer or set of layers) on top of these features to perform a specific task like classification, regression, etc.

#### Extracting Embeddings
The command provided demonstrates how to extract embeddings from the ProGen model using a dataset (for example, 'gb1'). The embeddings are extracted from the last layer of the model, and a mean reduction is applied to these embeddings:

```
python3 plmfit --function extract_embeddings \
                  --layer last --reduction mean --data_type gb1 \
                  --plm progen2-medium --output_dir $SCRATCH

```

- --function extract_embeddings: Specifies that the script should extract embeddings.
- --layer last: Indicates that embeddings should be extracted from the last layer of the model. You can specify other layers as needed.
- --reduction mean: Applies a mean reduction to the embeddings. This is useful for reducing the dimensionality of the embeddings for each sequence to a single vector.
- --data_type gb1: Specifies the dataset from which to extract embeddings.
- --plm progen2-medium: Specifies the pre-trained model to use for embedding extraction.
- --output_dir $SCRATCH: Specifies the directory where the extracted embeddings will be saved.


#### Transfer learning
After extracting embeddings, you can train a new model (or "head") on these embeddings to perform a specific task. The command provided demonstrates how to fine-tune a model with a new head for feature extraction:

```
python3 plmfit --function fine_tuning --ft_method feature_extraction \
                  --head mlp --head_config config_mlp.json \
                  --layer last --reduction mean --data_type gb1 --plm progen2-small \
                  --scaler standard --batch_size 64 --epochs 200
```

- --function fine_tuning: Specifies that the script should perform fine-tuning.
- --ft_method feature_extraction: Indicates that the fine-tuning method is based on feature extraction.
- --head mlp: Specifies the type of head to train. In this case, a multi-layer perceptron (MLP).
- --head_config config_mlp.json: Specifies the configuration file for the MLP head. This file contains details like the architecture of the MLP, including the number of layers and units in each layer.
- --layer last and --reduction mean: Similar to the embedding extraction command, these specify which layer's embeddings to use and the type of reduction.
- --data_type gb1 and --plm progen2-small: Specify the dataset and the pre-trained model to use.
- --scaler standard (Optional): Specifies the type of scaling to apply to the features before training the head. Standard scaling (z-score normalization) is commonly used.
- --batch_size 64 and --epochs 200: Specify the batch size and the number of epochs for training.

#### Task-Specific Head Concatenation (DEMONSTRATION ONLY)

You can concatenate a task-specific head to the model as follows:

```
python3 plmfit --function fine_tuning --ft_method feature_extraction \
                  --head_dir $SCRATCH --plm progen2-small
```

**Currently supports**
| **Transfer learning method**          | **Methods name**   | **Arguments** | **Relevant publication** | 
| ------------ | --------------------- | -------------- | ----------------- | 
| Full retraining | full-retrain           |   | data_type,plm   | NA    | 
| Feature extraction | feature-extraction       |   data_type,layer,reduction,plm,head,emb_dir |       | NA   | 

## Explainability analysis

| **Explainability analysis function**     | **Decription** | **Relevant publication** | **Publication date** |
| ------------ | --------------------- | -------------- | ----------------- | 
| TBD |       |    |      |     |

## Running on Euler

| PLM  version     | GPU size  (gres=gpumem: )  | RAM (mem-per-cpu) | 
| ------------  | -------------- | ----------------- |
| progen2-small    |   24g | 8000     | 
| progen2-medium       | 24g    | 12000       | 
| progen2-xlarge       | 80g    | 100000      | 
| ankh-base       | 40g    | 16000     | 
| esm2_t33_650M_UR50D | 40g    | 32000     | 
| antiberty | 16g    | 8000     | 
