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

4. **Install PLMFit**
   ```shell
   pip install .

## Usage

This section provides an overview of how to use the PLMFit package for various tasks.

**Currently supports**
| PLM          | Versions       | Publication Date | Source Link               | Owner        |
| ------------ | -------------- | ----------------- | ------------------------- | ------------ |
| ProGen  |    | 2022-01-01        |  | Salesforce     |
| ESM | esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D     | 2022-02-15        | [Source Link 1](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full), [Source Link 2](https://www.science.org/doi/10.1126/science.ade2574)   | Meta   |
| Ankh| ankh-base, ankh-large,ankh2-large     | 2023-01-18        | [Source Link 1](https://arxiv.org/abs/2301.06568)    | Proteinea   |
| ProGen  | progen2-small, progen2-medium, progen2-xlarge   | 2022-01-01        | [Source Link 1](https://github.com/salesforce/progen) | Salesforce     |

### Task-Specific Head Concatenation

You can concatenate a task-specific head to the model as follows (for demonstration purposes a simple LinearRegression head is being created):

```python
from models.models import LinearRegression
head = LinearRegression(input_dim=32, output_dim=1) 
model.concat_task_specific_head(head)
```
### Transfer learning

Fine-tuning allows you to train a PLM  for a specific task. Here's an example:
```
python3 plmfit.py --function fine-tuning --methods feature-extraction --layer last --reduction mean --data_type gb1 --plm progen2-medium --head linear --emb_dir $SCRATCH 
```

### Feature / Embeddings extraction

To extract embeddings or features from the model, you can use the following code:
(for demonstration purposes the ProGen embeddings (features) will be extracted from the 'aav' dataset from layer 11)

```
python3 plmfit.py --function extract_embeddings --layer last --reduction mean --data_type gb1 --plm progen2-medium --output_dir $SCRATCH

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
