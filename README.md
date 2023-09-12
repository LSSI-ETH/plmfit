# plmfit
PLMFIT is a framework that facilitates finetuning of  protein Language Models (currently supporting ProGen and ESM) on custom/experimental sequence data with one of the following methods 1. full retraining 2. ulmfit 3. feature extraction 4. lightweight adapters.

# Data

Each data type (i.e. protein, protein families, enrichments etc) should have its own folder named {data_type} (i.e. "gb1" for the gb1 protein). The sequence data for each data type should be saved as a .csv file with columns standarized and name {data_type}_data_full.csv. Specifically each data type folder should contain: 

1) a {data_type}_data_full.csv : File to save the aa sequences with standarized columns naming and structure. Column names **"aa_seq"** : includes AA sequences **"len"**: includes the length of the AA sequence **"score1"**,**"score2"** : includes a fitness score or generally a labels for the specific AA sequence (this values is data type and experiment that yielded the sequences dependant  and **"{split_name1}"**,**"{split_name2}"** : includes value 'train' , 'test' and 'valid' to facilitate the dataset splits during experiments.
