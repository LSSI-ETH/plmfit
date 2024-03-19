import json
import argparse
import itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics.functional as functional
import matplotlib as plt
import numpy as np
from plmfit.shared_utils import utils
import plmfit.models.downstream_heads as heads
import plmfit.shared_utils.data_explore as data_explore
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_name',type = str )

args = parser.parse_args()
exp_name = args.exp_name
exp_dir = "/cluster/scratch/bsahin/fine_tuning/"
json_name = exp_dir + exp_name + '/' + exp_name + '_data.json'

if __name__ == '__main__':

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # Load the experiment data from the json file
    with open(json_name, "r") as json_file:
        exp_data = json.load(json_file)

    # Load the arguments that were used to train the model   
    args = exp_data["arguments"]

    # Load data & embeddings
    data = utils.load_dataset(args["data_type"])
    embeddings = utils.load_embeddings(emb_path=f'{args["output_dir"]}/extract_embeddings/',
                                       data_type=args["data_type"], model=args["plm"], layer=args["layer"], 
                                       reduction=args["reduction"])

    print(embeddings.size())
    
    # Read head_config to initialize the model
    head_config = utils.load_head_config(args["head_config"])
    network_type = head_config['architecture_parameters']['network_type']

    if network_type == 'linear':
        head_config['architecture_parameters']['input_dim'] = embeddings.shape[1]
        pred_model = heads.LinearHead(head_config['architecture_parameters'])
    elif network_type == 'mlp':
        head_config['architecture_parameters']['input_dim'] = embeddings.shape[1]
        pred_model = heads.MLP(head_config['architecture_parameters'])
    else:
        raise ValueError('Head type not supported')

    # Load the trained model
    pred_model.load_state_dict(torch.load(args["experiment_dir"] + '/' + args["experiment_name"] + '.pt', map_location = device))

    # Test Dataset and dataloader is created
    data_test = data[data['random'] == 'test']
    embs_test = embeddings[data_test.index]

    del data
    del embeddings

    batch_size = 1000

    test_dataset = TensorDataset(
        embs_test, torch.tensor(data_test[["mouse","cattle","bat"]].values,dtype = torch.float))

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    y_pred, y_test = data_explore.test_multi_label_classification(pred_model, {"test":test_dataloader}, device)
    res, pool, plots = data_explore.evaluate_predictions(y_pred,y_test,"multilabel",n_class = 3)

    # Create dataframes to save results in
    results = pd.DataFrame(res).rename(index={0:"Mouse",1:"Cattle",2:"Bat"})
    pooled_results = pd.DataFrame(pool,index=[0]).rename(index={0:'Score'})

    # Export results to CSV files and save the plots
    plot_dir = result_dir = args["experiment_dir"] + "/plots"
    result_dir = args["experiment_dir"] + "/results"
    os.makedirs(result_dir, exist_ok = True)
    os.makedirs(plot_dir, exist_ok = True)

    # Save the results and figures
    results.to_csv(result_dir + '/results.csv')
    pooled_results.to_csv(result_dir + '/results_pooled.csv')
    for (name,figure) in plots.items():
        figure.savefig(plot_dir + f"/{name}.png")