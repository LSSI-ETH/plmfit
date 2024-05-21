# PLMFit

PLMFit is a powerful framework designed to democratize the fine-tuning of Protein Language Models (PLMs) for researchers with varying levels of computational expertise. With PLMFit, you can fine-tune state-of-the-art models on your experimental data through simple command-line instructions. This tool is particularly valuable for laboratory researchers seeking to leverage deep learning without needing in-depth programming knowledge. PLMFit also includes SCRUM scripts optimized for Euler systems, ensuring seamless integration and efficient execution of computational tasks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

### Prerequisites
Before you start, make sure Python 3.10 or higher is installed on your system. It is also recommended to manage your Python dependencies with a virtual environment to prevent conflicts with other packages.

### Steps to Install

1. **Clone the Repository:**
   Access the PLMFit repository within the lab's network and clone it to your local machine:
   ```bash
   git clone https://github.com/LSSI-ETH/plmfit.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd plmfit
   ```

3. **Create and Activate a Virtual Environment:**
   - For Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS and Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - For Scrum (Euler):
     A virtual environment is not needed, as there is one already at place. Load a python module to subsequently be able to install PLMFit.
     ```bash
     module load gcc/8.2.0 python_gpu/3.11.2
     ```

4. **Install PLMFit:**
   Install PLMFit using pip within your virtual environment:
   ```bash
   pip install -e .
   ```

### Configuration

Configure the `.env` file in the root directory as follows:

For local setups:
```
DATA_DIR='./plmfit'
```

For Euler and SCRUM an absolute path is required:
```
DATA_DIR='/absolute/path/to/plmfit'
```

For detailed data structure and setup, refer to the [data management guide](./plmfit/data/README.md).

## Supported PLMs
| Arguments | Model Name | Parameters | Source |
|-----------|------------|-------------|-------------|
| `esm2_t6_8M_UR50D` | ESM | 8M | [esm](https://github.com/facebookresearch/esm) |
| `esm2_t33_650M_UR50D` | ESM | 650M | [esm](https://github.com/facebookresearch/esm) |
| `esm2_t36_3B_UR50D` | ESM | 3B | [esm](https://github.com/facebookresearch/esm) |
| `esm2_t48_15B_UR50D` | ESM | 15B | [esm](https://github.com/facebookresearch/esm) |
| `progen2-small` | ProGen2 | 151M | [progen](https://github.com/enijkamp/progen2) |
| `progen2-medium` | ProGen2 | 764M | [progen](https://github.com/enijkamp/progen2) |
| `progen2-xlarge` | ProGen2 | 6.4B | [progen](https://github.com/enijkamp/progen2) |
| `proteinbert` | ProteinBERT | 94M | [proteinbert](https://github.com/nadavbra/protein_bert) |

## Usage

PLMFit facilitates easy application of protein language models (PLMs) for embedding extraction, fine-tuning, and other machine learning tasks through a user-friendly command-line interface. Below are detailed instructions for using PLMFit to perform various tasks:

### Extracting Embeddings

To extract embeddings from supported PLMs, use the following command structure:

```bash
python3 plmfit --function extract_embeddings \
               --data_type <dataset_short_name> \
               --plm <model_name> \
               --output_dir <output_directory> \
               --experiment_dir <experiment_directory> \
               --experiment_name <name_of_experiment> \
               --layer <layer_option> \
               --reduction <reduction_method>
```

**Parameters Explained:**
- `--function extract_embeddings`: Initiates the embedding extraction process.
- `--data_type`: Short name for the data to be used, as per naming conventions in README.
- `--plm`: Specifies the pre-trained model from supported PLMs.
- `--output_dir`: Directory for output, required for using the Euler SLURM scripts.
- `--experiment_dir`: Directory where experiment output files will be stored.
- `--experiment_name`: A unique name for identifying the experiment.
- `--layer`: (Optional) Specifies the model layer from which to extract embeddings ('first', 'quarter1', 'middle', 'quarter3', 'last'—default, or a specific layer number).
- `--reduction`: (Optional) Pooling method for embeddings ('mean'—default, 'bos', 'eos').

The output from the embedding extraction is a .pt file (PyTorch tensor) which contains the numerical representations of the sequences. Each sequence is transformed into an embedding vector, and the file size is determined by the number of sequences and the embedding size, essentially forming a matrix of size Sequences length X Embedding size. This structured data can then be used directly for machine learning models, providing a powerful toolset for predictive analytics and further research.

**Why Extract Embeddings?**
Extracting embeddings from protein sequences is a foundational step in bioinformatics. It converts complex protein sequences into a simpler, numerical format that machine learning models can easily process. By doing so, researchers can capture the intrinsic properties of proteins in a way that highlights their biological functionalities and interactions. This process is particularly useful for tasks such as protein classification, structure prediction, and function annotation.

### Fine-Tuning Models

Fine-tune supported PLMs using various techniques with the following command:

```bash
python3 -u plmfit --function fine_tuning \
                  --ft_method <fine_tuning_method> \
                  --target_layers <layer_targeting_option> \
                  --head_config <head_configuration_file> \
                  --data_type <dataset_short_name> \
                  --split <dataset_split> \
                  --plm <model_name> \
                  --output_dir <output_directory> \
                  --experiment_dir <experiment_directory> \
                  --experiment_name <name_of_experiment>
```

**Fine-Tuning Methods Explained:**
- `--ft_method`: Specifies the fine-tuning method ('feature_extraction', 'full', 'lora', 'bottleneck_adapters').
- `--target_layers`: Targets specific layers ('all' or 'last'), not applicable for 'feature_extraction'.
- `--head_config`: JSON configuration file for the head, defining the task (regression, classification, domain adaptation).
- Additional parameters similar to embedding extraction command.

**Understanding Fine-Tuning Methods:**
1. Feature Extraction:
   Description: This method involves extracting embeddings with a pre-trained model before fine-tuning a new head on these embeddings. It is less computationally intensive as it does not require updating the weights of the pre-trained model.
   Prerequisite: Embedding extraction must be completed first, as it uses these embeddings as input.
   Pros: Efficient in terms of computation; reduces the risk of overfitting on small datasets.
   Cons: May not capture as complex patterns as methods that update deeper model layers.
2. Full Fine-Tuning:
   Description: The layers of the model are updated during training. This method is suitable for tasks where the new dataset is large and significantly different from the data the model was initially trained on.
   Pros: Can significantly improve model performance on the task-specific data.
   Cons: Requires more computational resources; higher risk of overfitting on small datasets.
3. LoRA (Low-Rank Adaptation):
   Description: Modifies only a small part of the model's weights in a low-rank format, reducing the number of parameters that need to be updated.
   Pros: Less resource-intensive compared to full fine-tuning; can be effective even with smaller amounts of training data.
   Cons: Might not capture as wide a range of adaptations as full fine-tuning.
4. Bottleneck Adapters:
   Description: Introduces small bottleneck layers within the model that are trained while keeping the majority of the model's weights fixed.
   Pros: Allows for more targeted model updates without the need for extensive retraining of the entire network.
   Cons: May require careful tuning of the bottleneck architecture to achieve desired improvements.

### Train One Hot Encoding Models

To train models using one-hot encoding, utilize:

```bash
python3 -u plmfit --function one_hot \
                  --head_config <head_configuration_file> \
                  --data_type <dataset_short_name> \
                  --split <dataset_split> \
                  --output_dir <output_directory> \
                  --experiment_dir <experiment_directory> \
                  --experiment_name <name_of_experiment>
```

### Upcoming Features

- **Predict or Generate from Existing Models**: Coming soon.
