import json
import os

n_heads_values = [8,14] #[8, 16, 32, 64]
n_layers_values = [2] #[6, 12, 24, 36]
emb_dim_values = [16]#[512, 1024, 2048]
d_hid_values = [32,64,128] #[512, 1024, 2048]
dropout_values = [0.2] #, 0.3, 0.4, 0.5
optim_values = ['adam'] #'sgd' 
#ed_cut_off_values = [5, 6, 7, 8, 9]
ed_cut_off = 5
lr_values = [1e-3]
batch_size_values = [16]#, 32, 64, 128] ## , 256 , 512
head_values = ["emb_sum"] #, "emb_mean", "bos"]

split_value = "random"
c = 0
for nhead in n_heads_values:
    for nlayer in n_layers_values:
        for emb in emb_dim_values:
            for hid in d_hid_values:
                for drp in dropout_values:
                    for lr in lr_values:
                        for head in head_values:
                            for batch_size in batch_size_values:
                                config = {
                                    "n_heads": nhead,
                                    "n_layers": nlayer,
                                    "emb_dim": emb,
                                    "d_hid": hid,
                                    "dropout": drp,
                                    "ed_cut_off": ed_cut_off,
                                    "lr": lr,
                                    "batch_size": batch_size,
                                    "head":head,
                                    "optimizer":"adam",
                                    "sampler": "weighted",
                                    "warmup": 0,
                                    "scheduler": "attn_original"
                                }
                                config_name = f"experiment_config_{c}"
                                experiment_dir = f'./experiments/{config_name}'
                                if not os.path.exists(experiment_dir):
                                    os.makedirs(experiment_dir)
                                    
                                output_json = f"{experiment_dir}/{config_name}.json"
                                c += 1
                                # Save the dictionary as a JSON file
                                with open(output_json, 'w') as json_file:
                                    json.dump(config, json_file, indent=2)





print(c)
print(f"Dictionaries saved as JSON in 'experiments' folder")

