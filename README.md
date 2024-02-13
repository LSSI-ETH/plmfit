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
| PLM         | Versions       | Publication Date | Source Link               | Owner        |
| ------------  | -------------- | ----------------- | ------------------------- | ------------ |
| ProGen         |    | 2022-01-01        |  | Salesforce     |
| ESM         | v2.0, v2.1     | 2022-02-15        | [Source Link 2](link2)    | Meta   |


### Transfer learning

Fine-tuning allows you to train a PLM  for a specific task. Here's an example:
(for demonstration purposes the model will be fully_retrained ("full_retrain") on the 'aav' dataset with the correspoding labels)

```
python3 plmfit.py --function fine-tuning --layer last --reduction mean --data_type gb1 --plm progen2-medium --emb_dir $SCRATCH --head linear

```


### Feature / Embeddings extraction

To extract embeddings or features from the model, you can use the following code:
(for demonstration purposes the ProGen embeddings (features) will be extracted from the 'aav' dataset from layer 11)

```
python3 plmfit.py --function extract_embeddings --layer last --reduction mean --data_type gb1 --plm progen2-medium --output_dir $SCRATCH

```
Adjust the `dataset_name`, `batch_size`, and `layer` parameters as needed for your specific use case. (See supported data_types)
**Currently supports**
| **Transfer learning method**          | **Methods name**   | **Arguments** | **Relevant publication** | 
| ------------ | --------------------- | -------------- | ----------------- | 
| Full retraining | full-retrain           |   | data_type,plm   | NA    | 
| Feature extraction | feature-extraction       |   data_type,layer,reduction,plm,head,emb_dir |       | NA   | 


Adjust the `dataset_name`, `batch_size`, and `layer` parameters as needed for your specific use case. (See supported data_types and fine_tuning_mode)
## Explainability analysis

| **Explainability analysis function**     | **Decription** | **Relevant publication** | **Publication date** |
| ------------ | --------------------- | -------------- | ----------------- | 
| TBD |       |    |      |     |

## Running on Euler

