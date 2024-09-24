# PLMFit

PLMFit is a powerful framework designed to democratize the fine-tuning of Protein Language Models (PLMs) for researchers with varying levels of computational expertise. With PLMFit, you can fine-tune state-of-the-art models on your experimental data through simple command-line instructions. This tool is particularly valuable for laboratory researchers seeking to leverage deep learning without needing in-depth programming knowledge. PLMFit also includes SCRUM scripts optimized for Euler, the ETH supercomputer, ensuring seamless integration and efficient execution of computational tasks.

## Table of contents

- [Installation](#installation)
- [Usage](#usage)
- [Scoreboard](#scoreboard)
- [Contributions](#contributions)

## Installation

### Prerequisites
Before you start, make sure Python 3.10 or higher is installed on your system. It is also recommended to manage your Python dependencies with a virtual environment to prevent conflicts with other packages.

### Steps to install

1. **Clone the repository:**
   Access the PLMFit repository within the lab's network and clone it to your local machine:
   ```bash
   git clone https://github.com/LSSI-ETH/plmfit.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd plmfit
   ```

3. **Create and activate a virtual environment:**
   - For Windows:
      ```bash
      python3 -m venv venv
      venv\Scripts\activate
      ```
   - For macOS and Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
   - For SCRUM setups (Euler):
      Load a python module to subsequently be able to install PLMFit. For example, in ETH Euler Cluster:
      ```bash
      module load stack/2024-06 gcc/12.2.0
      module load python/3.11.6
      python3 -m venv venv
      source venv/bin/activate
      ```

4. **Install PLMFit:**
   Install PLMFit using pip within your virtual environment:
   ```bash
   pip install -e .
   ```

### Configuration

Configure the `.env` file in the root directory as follows:

For local setups only the data and config folder paths need to be defined:
```
DATA_DIR='./data'
CONFIG_DIR='./config'
```

For Euler and SCRUM an absolute path is required. To use the SCRUM scripts, the username and virtual environment need to be defined as well:
```
DATA_DIR='/absolute/path/to/plmfit'
CONFIG_DIR='/absolute/path/to/config'
SLURM_USERNAME='slurm_username'
VIRTUAL_ENV='/absolute/path/to/venv'
```

For detailed data structure and setup, refer to the [data management guide](./data/README.md).

## Supported PLMs
| Arguments | Model Name | Parameters | No. of Layers | Embedding dim. | Source |
|-----------|------------|-------------|-------------|-------------|-------------|
| `esm2_t6_8M_UR50D` | ESM | 8M | 6 | 320 | [esm](https://github.com/facebookresearch/esm) |
| `esm2_t33_650M_UR50D` | ESM | 650M  | 33 | 1280 | [esm](https://github.com/facebookresearch/esm) |
| `esm2_t36_3B_UR50D` | ESM | 3B | 36 | 2560 | [esm](https://github.com/facebookresearch/esm) |
| `esm2_t48_15B_UR50D` | ESM | 15B | 48 | 5120 | [esm](https://github.com/facebookresearch/esm) |
| `progen2-small` | ProGen2 | 151M | 11 | 1024 | [progen](https://github.com/enijkamp/progen2) |
| `progen2-medium` | ProGen2 | 764M | 26 | 1536 | [progen](https://github.com/enijkamp/progen2) |
| `progen2-xlarge` | ProGen2 | 6.4B | 32 | 4096 | [progen](https://github.com/enijkamp/progen2) |
| `proteinbert` | ProteinBERT | 92M | 11 | 768 | [proteinbert](https://github.com/nadavbra/protein_bert) |

## Usage

PLMFit facilitates easy application of Protein Language Models (PLMs) for embedding extraction, fine-tuning, and other machine learning tasks through a user-friendly command-line interface. Below are detailed instructions for using PLMFit to perform various tasks:

### Extracting embeddings

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

**Parameters explained:**
- `--function extract_embeddings`: Initiates the embedding extraction process.
- `--data_type`: Short name for the data to be used, as per naming conventions in README.
- `--plm`: Specifies the pre-trained model from supported PLMs.
- `--output_dir`: Directory for output, required for using the Euler SLURM scripts.
- `--experiment_dir`: Directory where experiment output files will be stored.
- `--experiment_name`: A unique name for identifying the experiment.
- `--layer`: (Optional) Specifies the model layer from which to extract embeddings ('first', 'quarter1', 'middle', 'quarter3', 'last'—default, or a specific layer number).
- `--reduction`: (Optional) Pooling method for embeddings ('mean'—default, 'bos', 'eos').

The output from the embedding extraction is a .pt file (PyTorch tensor) which contains the numerical representations of the sequences. Each sequence is transformed into an embedding vector, and the file size is determined by the number of sequences and the embedding size, essentially forming a matrix of size Sequences length X Embedding size. This structured data can then be used directly for machine learning models, providing a powerful toolset for predictive analytics and further research.

**Why Extract embeddings?**
Extracting embeddings from protein sequences is a foundational step in bioinformatics. It converts complex protein sequences into a simpler, numerical format that machine learning models can easily process. By doing so, researchers can capture the intrinsic properties of proteins in a way that highlights their biological functionalities and interactions. This process is particularly useful for tasks such as protein classification, structure prediction, and function annotation.

### Fine-Tuning models

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
- `--head_config`: JSON configuration file for the head, defining the task (regression, classification, domain adaptation). This JSON file needs to be located in `./config/training/` folder. The argument should be the relative path of the file to the `./config/training/` folder. For further documentation on how the head should be structured, refer to the [training management guide](./config/training/README.md).

**Understanding Fine-Tuning Methods:**
1. **Feature Extraction:**
   - Description: This method involves extracting embeddings with a pre-trained model before fine-tuning a new head on these embeddings. It is less computationally intensive as it does not require updating the weights of the pre-trained model. To automatically use embeddings extracted beforehand, use the same `output_dir` argument.
   - Prerequisite: Embedding extraction must be completed first, as it uses these embeddings as input. The same `output_dir` argument needs to be passed.
   - Pros: Efficient in terms of computation; reduces the risk of overfitting on small datasets.
   - Cons: May not capture as complex patterns as methods that update deeper model layers.
2. **Full Fine-Tuning:**
   - Description: The layers of the model are updated during training. This method is suitable for tasks where the new dataset is large and significantly different from the data the model was initially trained on.
   - Pros: Can significantly improve model performance on the task-specific data.
   - Cons: Requires more computational resources; higher risk of overfitting on small datasets.
3. **LoRA (Low-Rank Adaptation):**
   - Description: Modifies only a small part of the model's weights in a low-rank format, reducing the number of parameters that need to be updated.
   - Pros: Less resource-intensive compared to full fine-tuning; can be effective even with smaller amounts of training data.
   - Cons: Might not capture as wide a range of adaptations as full fine-tuning.
4. **Bottleneck Adapters:**
   - Description: Introduces small bottleneck layers within the model that are trained while keeping the majority of the model's weights fixed.
   - Pros: Allows for more targeted model updates without the need for extensive retraining of the entire network.
   - Cons: May require careful tuning of the bottleneck architecture to achieve desired improvements.

**Advanced usage:**
You can change the configuration of LoRA and Bottleneck Adapters by adapting the relevant config file found in `./config/peft/` folder. Change these parameters only if you have experience with these methods or want to experiment with different settings.

### Train One-Hot Encoding models

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

### Using PLMFit on a SCRUM setup (e.g. Euler)
Navigate to the `scripts` folder, where you will find subfolders for each of the platform's features. Adjust the `experiments_setup.csv` file according to your needs and simply call `./scripts/{function}/submit_{function}_mass.sh` from the parent directory. The columns in this file represent various arguments, most of which are the same as those mentioned previously. Here are the key columns:

- `gpus`: The number of GPUs to request.
- `gres`: The type of GPU to request, either by name or by size.
- `mem-per-cpu`: The amount of CPU RAM to allocate per GPU.
- `nodes`: The number of nodes to request.
- `run_time`: The duration for which the job should run in hours.
- `experimenting`: Set this if you want to benchmark speed, resource usage, etc.

Use tabs as deliminators and the last line has to stay blank, otherwise the scripts will not function.

### Upcoming features

- **Predict or generate from existing models**: Coming soon.

## Scoreboard
| Task | Score | Metric | PLM | TL method | Layers used | Pooling | Downstream head |
|-----------|------------|-------------|-------------|-------------|-------------|-------------|-------------|
| AAV - sampled | 0.932 | Spearman's | ESM2-15B | Adapters | All | Mean | Linear |
| AAV - one-vs-rest | 0.831 | Spearman's | ProGen2-XL  | LoRA | 75% | CLS | Linear |
| GB1 - three-vs-rest | 0.879 | Spearman's | ProGen2-M | Adapters | 50% | CLS | Linear |
| GB1 - one-vs-rest | 0.457 | Spearman's | ProGen2-S | FE | 75% | Mean | Linear |
| Meltome - mixed | 0.723 | Spearman's | ProGen2-XL | LoRA | All | Mean | Linear |
| HER2 - one-vs-rest | 0.390 | MCC | ProGen2-S | LoRA- | 50% | CLS | Linear |
| RBD - one-vs-rest | 0.554 | MCC | ProGen2-S | LoRA | 50% | Mean | Linear |

## Contributions 🎉
We welcome contributions from the community! If you're interested in contributing to PLMFit, feel free to:

*   **Expand the benchmarked datasets**: We invite you to add new datasets to our benchmarks or experiment with different setups for existing datasets. Your contributions will help improve the robustness and versatility of PLMFit.

*   **Submit a pull request (PR)**: Whether it's a bug fix, a new feature, or an improvement, we encourage you to submit a PR.
    
*   **Contact us directly**: If you have any questions or need guidance on how to contribute, don't hesitate to reach out to us.
    
*   **Open issues**: For developers interested in contributing, you can open issues to report bugs, suggest new features, or discuss potential improvements.
        
Your contributions are highly valued and will help us enhance PLMFit for everyone. Thank you for your interest and support!

## Citations
If you found PLMFit useful for your research, we ask you to cite the paper:
```
@article{bikias2024plmfit,
  author={Bikias, Thomas and Stamkopoulos, Evangelos and Reddy, Sai},
  title={PLMFit: Benchmarking Transfer Learning with Protein Language Models for Protein Engineering},
  year={2024},
  doi={tbd},
  url={tbd},
  journal={tbd}
}
```

## License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.