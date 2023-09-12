# plmfit
PLMFIT is a framework that facilitates finetuning of  protein Language Models (currently supporting ProGen and ESM) on custom/experimental sequence data with one of the following methods 1. full retraining 2. ulmfit 3. feature extraction 4. lightweight adapters.

# Data

Each data type (i.e. protein, protein families, enrichments etc) should have its own folder named {data_type} (i.e. "gb1" for the gb1 protein). The sequence data for each data type should be saved as a .csv file with columns standarized and name {data_type}_data_full.csv. Specifically each data type folder should contain: 

1) a {data_type}_data_full.csv : File to save the aa sequences with standarized columns naming and structure.<br />
   <ins>Column names</ins><br />
                **"aa_seq"** : includes AA sequences <br />
                **"len"**: includes the length of the AA sequence<br />
                **"score1"**,**"score2"** : includes a fitness score or generally a labels for the specific AA sequence (this values is data type and task dependant <br />
                **"{split_name1}"**,**"{split_name2}"** : includes value 'train' , 'test' and 'valid' to facilitate the dataset splits during experiments.<br />

2) data_parse.py : a python script that parses the data corresponding to the data type. Performs data preprocessing (duplicates removal, denoising) and formats the data to produce the {data_type}_data_full.csv

3) "embeddings" folder : Includes .pt files named "{data_type}_{model_name}_embs_layer{layer}_{reduction}.pt" corresponding to the protein language model, the layer and the reduction used to calculate the embeddings for the amino acid sequences of the respective data type

