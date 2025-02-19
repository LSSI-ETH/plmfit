import os
from plmfit.language_models.progen2.models.progen.modeling_progen import (
    ProGenForSequenceClassification,
    ProGenForEmbeddingsExtraction,
    ProGenForTokenClassification,
)
from plmfit.language_models.proteinbert.modeling_bert import (
    ProteinBertForSequenceClassification,
    ProteinBertForMaskedLM,
    ProteinBertForEmbeddingsExtraction,
    ProteinBertForTokenClassification,
)
from plmfit.language_models.esm.modeling_esm import (
    PlmfitEsmForSequenceClassification,
    PlmfitEsmForMaskedLM,
    PlmfitEsmForTokenClassification,
    PlmfitEsmForEmbdeddingsExtraction,
)
from plmfit.language_models.esmC.modeling_esmc import (
    PlmfitEsmCForSequenceClassification,
    PlmfitEsmCForEmbdeddingsExtraction,
)

# from plmfit.shared_utils.data_explore import visualize_embeddings

from plmfit.shared_utils.linear_block import ProGenLinearBlock
import plmfit.shared_utils.utils as utils
import torch.nn as nn
import plmfit.logger as l
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import time
from abc import abstractmethod
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    AutoModel,
    EsmForMaskedLM,
    EsmForSequenceClassification,
)
from antiberty import AntiBERTyRunner
from numpy import array
import psutil
import traceback
import torch.nn.functional as F


class IPretrainedProteinLanguageModel(nn.Module):

    name: str
    py_model: nn.Module
    head: nn.Module
    head_name: str
    no_parameters: int
    emb_layers_dim: int
    output_dim: int
    logger: l.Logger

    def __init__(self, logger, task="regression"):
        super().__init__()
        self.head_name = "none"
        self.logger = logger
        self.task = task

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_py_model(self):
        return self.py_model

    def set_py_model(self, py_model):
        self.py_model = py_model

    def get_tokenizer(self):
        return self.tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_layer_to_use(self, layer):
        if layer == "last":
            # The last hidden layer
            self.layer_to_use = self.no_layers - 1
        elif layer == "middle":
            self.layer_to_use = (self.no_layers - 1) // 2
        elif layer == "first":
            self.layer_to_use = 0
        elif layer == "quarter1":
            self.layer_to_use = (self.no_layers - 1) // 4
        elif layer == "quarter3":
            self.layer_to_use = (self.no_layers - 1) // 2 + (self.no_layers - 1) // 4
        else:
            # Fallback for numeric layer specification or unexpected strings
            self.layer_to_use = int(layer) if layer.isdigit() else self.no_layers - 1

        self.py_model.trim_model(self.layer_to_use)


class Antiberty(IPretrainedProteinLanguageModel):
    def __init__(self):
        super().__init__()
        self.name = "antiberty"
        self.logger = l.Logger(f"{self.name}")
        self.model = AntiBERTyRunner()

    def extract_embeddings(self, data_type, layer, reduction, output_dir="default"):
        logger = self.logger
        device = "cpu"
        fp16 = False
        device_ids = []
        if torch.cuda.is_available():
            device = "cuda:0"
            fp16 = True
            self.logger.log(f"Available GPUs : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.log(
                    f"Running on {torch.cuda.get_device_properties(i).name}"
                )
                device_ids.append(i)

        else:
            self.logger.log(f"No gpu found rolling device back to {device}")

        batch_size = 1
        if output_dir == "default":
            output_path = f"./plmfit/data/{data_type}/embeddings/"
        else:
            output_path = f"{output_dir}/{data_type}/embeddings/"
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        start_emb_time = time.time()
        data = utils.load_dataset(data_type)

        self.logger = l.Logger(
            f"extract_embeddings_{data_type}_{self.name}_{reduction}"
        )

        embs = torch.zeros((len(data), 1024)).to(device)

        for vh, vl in zip(data["V_h"], data["V_l"]):
            start = time.time()
            out = self.model.embed([vh, vl])
            if reduction == "mean":
                embs[i, 0:512] = torch.mean(out[0], dim=0)
                embs[i, 512:1024] = torch.mean(out[1], dim=0)
            elif reduction == "sum":
                embs[i, 0:512] = torch.sum(out[0], dim=0)
                embs[i, 512:1024] = torch.sum(out[1], dim=0)
            elif reduction == "bos":
                # Select the embeddings for the first token of each sequence in the batch
                embs[i, 0:512] = out[0][0, :]
                embs[i, 512:1024] = out[1][0, :]
            elif reduction == "eos":
                # Select the embeddings for the last token of each sequence in the batch
                embs[i, 0:512] = out[0][-1, :]
                embs[i, 512:1024] = out[1][-1, :]
            else:
                raise ValueError("Unsupported reduction option")
            del out
            i = i + batch_size
            logger.log(f" {i} / {len(data)} | {time.time() - start:.2f}s ")

        torch.save(
            embs,
            os.path.join(
                output_path, f"{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt"
            ),
        )
        t = torch.load(
            os.path.join(
                output_path, f"{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt"
            )
        )
        logger.log(
            f'Saved embeddings ({t.shape[1]}-d) as "{data_type}_{self.name}_embs_layer{layer}_{reduction}.pt" ({time.time() - start_emb_time:.2f}s)'
        )
        return


class ProGenFamily(IPretrainedProteinLanguageModel):

    tokenizer: Tokenizer

    def __init__(
        self, progen_model_name: str, logger: l.Logger, task: str = "regression"
    ):
        super().__init__(logger)
        self.name = progen_model_name
        self.task = task
        if self.task == "causal_lm":
            raise ValueError("Causal LM not supported yet for ProGen")
            # self.py_model : PlmfitEsmForMaskedLM = PlmfitEsmForMaskedLM.from_pretrained(f'facebook/{esm_version}', output_hidden_states = True)
            # self.output_dim = self.py_model.lm_head.decoder.out_features
        elif self.task == "token_classification":
            self.py_model = ProGenForTokenClassification.from_pretrained(
                f"{utils.plmfit_path}/language_models/progen2/checkpoints/{progen_model_name}"
            )
            self.output_dim = self.py_model.classifier.out_features
        elif self.task == "extract_embeddings":
            self.py_model = ProGenForEmbeddingsExtraction.from_pretrained(
                f"{utils.plmfit_path}/language_models/progen2/checkpoints/{progen_model_name}"
            )
        else:
            self.py_model: ProGenForSequenceClassification = (
                ProGenForSequenceClassification.from_pretrained(
                    f"{utils.plmfit_path}/language_models/progen2/checkpoints/{progen_model_name}"
                )
            )
            self.output_dim = self.py_model.classifier.out_features
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.transformer.h)
        self.emb_layers_dim = self.py_model.transformer.h[0].attn.out_proj.out_features
        self.tokenizer = utils.load_tokenizer(progen_model_name)
        self.layer_to_use = self.no_layers - 1
        self.config = self.py_model.config
        self.experimenting = False

    def zeroed_model(self):
        # Neutralize the model by setting weights to ones and bias to zeros
        with torch.no_grad():
            self.py_model.transformer.h = nn.ModuleList(
                [ProGenLinearBlock(self.config) for _ in range(self.config.n_layer)]
            )
            self.py_model.transformer.ln_f.weight.fill_(1.0)
            self.py_model.transformer.ln_f.bias.fill_(0.0)
            self.logger.log("Zeroed the model with plain linear blocks")

    def categorical_encode(self, data, max_length="default"):
        encs = utils.categorical_encode(
            data["aa_seq"].values,
            self.tokenizer,
            max(data["len"].values) if max_length == "default" else max_length,
            add_bos=True,
            add_eos=True,
            logger=self.logger,
        )
        return encs

    def extract_embeddings(
        self, data_type, batch_size=1, layer=11, reduction="mean", log_interval=1000
    ):
        try:
            self.set_layer_to_use(layer)
            layer = (
                self.layer_to_use + 1
            )  # Adjusted for considering input embeddings as well
            self.logger.log(f"Extracting embeddings from layer: {layer}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            memory_usage = psutil.virtual_memory()
            max_mem_usage = utils.print_gpu_utilization(memory_usage, device)
            fp16 = False
            device_ids = []
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(1 / 1)
                torch.cuda.memory._record_memory_history(
                    enabled="all", max_entries=100000
                )
                device = "cuda:0"
                fp16 = True
                self.logger.log(f"Available GPUs : {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.logger.log(
                        f"Running on {torch.cuda.get_device_properties(i).name}"
                    )
                    device_ids.append(i)

            else:
                self.logger.log(f"No gpu found rolling device back to {device}")

            data = utils.load_dataset(data_type)

            start_enc_time = time.time()
            self.logger.log(f"Encoding {data.shape[0]} sequences....")
            encs = self.categorical_encode(data)
            mem_usage = utils.print_gpu_utilization(memory_usage, device)
            if mem_usage > max_mem_usage:
                max_mem_usage = mem_usage
            self.logger.log(f"Encoding completed! {time.time() -  start_enc_time:.4f}s")
            enc_time = time.time() - start_enc_time
            encs = encs.to(device)
            seq_dataset = data_utils.TensorDataset(encs)
            seq_loader = data_utils.DataLoader(
                seq_dataset, batch_size=batch_size, shuffle=False
            )
            self.logger.log(
                f"Extracting embeddings for {len(seq_dataset)} sequences..."
            )

            # FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
            embs = torch.zeros((len(seq_dataset), self.emb_layers_dim)).to(device)
            self.py_model = self.py_model.to(device)
            i = 0
            self.py_model.eval()
            start_extraction_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16, cache_enabled=False):
                    for batch in seq_loader:
                        start = time.time()
                        current_batch_size = batch[0].shape[0]
                        if layer == "logits":
                            out = self.py_model(batch[0]).logits
                            mem_usage = utils.print_gpu_utilization(
                                memory_usage, device
                            )
                            if mem_usage > max_mem_usage:
                                max_mem_usage = mem_usage
                        else:
                            model_output = self.py_model(batch[0])
                            mem_usage = utils.print_gpu_utilization(
                                memory_usage, device
                            )
                            if mem_usage > max_mem_usage:
                                max_mem_usage = mem_usage
                            hidden_states = (
                                model_output.hidden_states
                            )  # Get all hidden states

                            # Log the shape of each layer's embeddings for the first batch if we are unsure what our model outputs
                            # if i == 0:
                            #     for layer_index, layer_output in enumerate(hidden_states):
                            #         self.logger.log(f'Layer {layer_index} shape: {layer_output.shape}')

                            # Now select the specific layer's output
                            out = hidden_states[layer]
                        if reduction == "mean":
                            embs[i : i + current_batch_size, :] = torch.mean(out, dim=1)
                            if i == 2:
                                self.logger.log(f"{(torch.mean(out, dim=1)).size()}")
                                # visualize_embeddings(torch.mean(out, dim=1), use_heatmap=False)
                        elif reduction == "weighted_mean":
                            # Calculate L2 norms of vectors across the feature dimension
                            l2_norms = torch.norm(out, p=2, dim=-1)

                            # Calculate variance of vectors across the feature dimension
                            variances = torch.var(out, dim=-1, unbiased=False)

                            # Normalize the outputs to positive values to calculate entropy
                            normalized_outputs = F.softmax(out, dim=-1)
                            # Calculate entropy across the feature dimension
                            entropy = -torch.sum(
                                normalized_outputs
                                * torch.log(normalized_outputs + 1e-10),
                                dim=-1,
                            )

                            # Apply softmax to L2 norms to get weights for each token
                            weights = F.softmax(l2_norms, dim=1)
                            print(weights)
                            # Use weights to perform weighted mean across the sequence length dimension
                            weighted_mean = torch.sum(
                                out * weights.unsqueeze(-1), dim=1
                            )
                            embs[i : i + current_batch_size, :] = weighted_mean
                            if i == 2:
                                # print(f"First batch weighted mean vector: {weighted_mean}")
                                self.logger.log(
                                    f"Weighted Mean of first batch: {weighted_mean.numpy()}"
                                )
                                # visualize_embeddings(weighted_mean, use_heatmap=False)

                        elif reduction == "sum":
                            embs[i : i + current_batch_size, :] = torch.sum(out, dim=1)
                        elif reduction == "bos":
                            # Select the embeddings for the first token of each sequence in the batch
                            embs[i : i + current_batch_size, :] = out[:, 0, :]
                        elif reduction == "eos":
                            # Initialize a tensor to store the selected embeddings
                            selected_embs = torch.zeros(
                                current_batch_size, out.shape[2], device=out.device
                            )

                            # Iterate over each sequence in the batch
                            for seq_idx in range(current_batch_size):
                                # Find the positions of the token with ID equal to 2 in the current sequence
                                token_positions = (batch[0][seq_idx] == 2).nonzero(
                                    as_tuple=True
                                )[0]

                                # Check if the token ID is present in the sequence
                                if len(token_positions) > 0:
                                    # Select the position of the last occurrence of the token
                                    last_position = token_positions[-1].item()

                                    # Select the embeddings for the last occurrence of the token
                                    selected_embs[seq_idx, :] = out[
                                        seq_idx, last_position, :
                                    ]
                                else:
                                    # If the token ID is not found, you might want to handle this case.
                                    # For example, use the embeddings of the last token of the sequence
                                    # or set to zeros, depending on your application's requirements.
                                    # Here, we use the embeddings of the last token as a fallback.
                                    selected_embs[seq_idx, :] = out[seq_idx, -1, :]
                                    self.logger.log(
                                        f"EOS token not found for sequence {i}"
                                    )

                            # Update the embeddings tensor with the selected embeddings
                            embs[i : i + current_batch_size, :] = selected_embs
                            if i == 2:
                                self.logger.log(selected_embs)
                                # visualize_embeddings(selected_embs, use_heatmap=False)
                        elif utils.convert_to_number(reduction) is not None:
                            # Select the embeddings for the i token of each sequence in the batch
                            embs[i : i + current_batch_size, :] = out[
                                :, utils.convert_to_number(reduction), :
                            ]
                        else:
                            raise ValueError("Unsupported reduction option")
                        del out
                        i = i + current_batch_size
                        if log_interval != -1 and i % log_interval == 0:
                            self.logger.log(
                                f" {i} / {len(seq_dataset)} | {time.time() - start:.2f}s "
                            )
            extraction_time = time.time() - start_extraction_time
            torch.save(embs, f"{self.logger.base_dir}/{self.logger.experiment_name}.pt")
            file_size_bytes = os.path.getsize(
                f"{self.logger.base_dir}/{self.logger.experiment_name}.pt"
            )
            file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
            report = {
                "encoding_time_needed": f"{enc_time:.4f}",
                "extraction_time_needed": f"{extraction_time:.4f}",
                "avg_time_per_seq": f"{extraction_time / len(seq_dataset):.4f}",
                "dataset_len": len(seq_dataset),
                "embeddings_dim": embs.shape,
                "data_type": data_type,
                "layer": layer,
                "reduction": reduction,
                "batch_size": batch_size,
                "embeddings_file_size_mb": f"{file_size_mb:.2f}",
                "max_vram_usage_mb": max_mem_usage,
            }
            self.logger.save_data(report, "report")
            t = torch.load(f"{self.logger.base_dir}/{self.logger.experiment_name}.pt")
            self.logger.log(
                f'Saved embeddings ({t.shape[1]}-d) as "{self.logger.experiment_name}.pt" ({time.time() - start_enc_time:.2f}s)'
            )
            if torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(
                    f"{self.logger.base_dir}/memory_profiler.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)
            return
        except:
            if torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(
                    f"{self.logger.base_dir}/memory_profiler.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)
            stack_trace = traceback.format_exc()
            self.logger.log(stack_trace)

    def evaluate(self):
        return 0

    def forward(self, src):
        src = self.py_model(src).hidden_states[self.layer_to_use]
        src = torch.mean(src, dim=1)
        if self.head != None:
            src = self.head(src)
        return src


class ESMFamily(IPretrainedProteinLanguageModel):
    tokenizer: AutoTokenizer

    def __init__(self, esm_version: str, logger: l.Logger, task: str = "regression"):
        super().__init__(logger, task)
        self.version = esm_version
        if self.task == "masked_lm":
            self.py_model: PlmfitEsmForMaskedLM = PlmfitEsmForMaskedLM.from_pretrained(
                f"facebook/{esm_version}", output_hidden_states=True
            )
            self.output_dim = self.py_model.lm_head.decoder.out_features
        elif self.task == "token_classification":
            self.py_model = PlmfitEsmForTokenClassification.from_pretrained(
                f"facebook/{esm_version}", output_hidden_states=True
            )
            self.output_dim = self.py_model.classifier.out_features
        elif self.task == "extract_embeddings":
            self.py_model = PlmfitEsmForEmbdeddingsExtraction.from_pretrained(
                f"facebook/{esm_version}", output_hidden_states=True
            )
        else:
            self.py_model = PlmfitEsmForSequenceClassification.from_pretrained(
                f"facebook/{esm_version}", output_hidden_states=True
            )
            self.output_dim = self.py_model.classifier.out_features
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.esm.encoder.layer)
        self.emb_layers_dim = self.py_model.esm.encoder.layer[
            0
        ].attention.self.query.in_features
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{esm_version}")
        self.layer_to_use = self.no_layers - 1
        self.experimenting = False

    def extract_embeddings(
        self, data_type, batch_size=1, layer=11, reduction="mean", log_interval=1000
    ):
        try:
            self.set_layer_to_use(layer)
            layer = (
                self.layer_to_use + 1
            )  # Adjusted for considering input embeddings as well
            self.logger.log(f"Extracting embeddings from layer: {layer}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            memory_usage = psutil.virtual_memory()
            max_mem_usage = utils.print_gpu_utilization(memory_usage, device)
            fp16 = False
            device_ids = []
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(1 / 1)
                torch.cuda.memory._record_memory_history(
                    enabled="all", max_entries=100000
                )
                device = "cuda:0"
                fp16 = True
                self.logger.log(f"Available GPUs : {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.logger.log(
                        f"Running on {torch.cuda.get_device_properties(i).name}"
                    )
                    device_ids.append(i)

            else:
                self.logger.log(f"No gpu found rolling device back to {device}")

            data = utils.load_dataset(data_type)
            # data = data.sample(10)
            start_enc_time = time.time()
            self.logger.log(f"Encoding {data.shape[0]} sequences....")
            encs = self.categorical_encode(data)
            mem_usage = utils.print_gpu_utilization(memory_usage, device)
            if mem_usage > max_mem_usage:
                max_mem_usage = mem_usage
            self.logger.log(f"Encoding completed! {time.time() -  start_enc_time:.4f}s")
            enc_time = time.time() - start_enc_time
            encs = encs.to(device)
            seq_dataset = data_utils.TensorDataset(encs)
            seq_loader = data_utils.DataLoader(
                seq_dataset, batch_size=batch_size, shuffle=False
            )
            self.logger.log(
                f"Extracting embeddings for {len(seq_dataset)} sequences..."
            )

            # FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
            embs = torch.zeros((len(seq_dataset), self.emb_layers_dim)).to(device)
            self.py_model = self.py_model.to(device)
            i = 0
            self.py_model.eval()
            start_extraction_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16, cache_enabled=False):
                    for batch in seq_loader:
                        start = time.time()
                        current_batch_size = batch[0].shape[0]
                        if layer == "logits":
                            out = self.py_model(batch[0]).logits
                        else:
                            model_output = self.py_model(batch[0])
                            hidden_states = model_output.hidden_states

                            # Log the shape of each layer's embeddings for the first batch if we are unsure what our model outputs
                            # if i == 0:
                            #     for layer_index, layer_output in enumerate(hidden_states):
                            #         self.logger.log(f'Layer {layer_index} shape: {layer_output.shape}')

                            # Now select the specific layer's output
                            out = hidden_states[layer]
                        if reduction == "mean":
                            embs[i : i + current_batch_size, :] = torch.mean(out, dim=1)
                            if i == 0:
                                self.logger.log(f"{(torch.mean(out, dim=1)).size()}")
                        elif reduction == "sum":
                            embs[i : i + current_batch_size, :] = torch.sum(out, dim=1)
                        elif reduction == "bos":
                            # Select the embeddings for the first token of each sequence in the batch
                            embs[i : i + current_batch_size, :] = out[:, 0, :]
                        elif reduction == "eos":
                            # Initialize a tensor to store the selected embeddings
                            selected_embs = torch.zeros(
                                current_batch_size, out.shape[2], device=out.device
                            )

                            # Iterate over each sequence in the batch
                            for seq_idx in range(current_batch_size):
                                # Find the positions of the token with ID equal to 2 in the current sequence
                                token_positions = (batch[0][seq_idx] == 2).nonzero(
                                    as_tuple=True
                                )[0]

                                # Check if the token ID is present in the sequence
                                if len(token_positions) > 0:
                                    # Select the position of the last occurrence of the token
                                    last_position = token_positions[-1].item()

                                    # Select the embeddings for the last occurrence of the token
                                    selected_embs[seq_idx, :] = out[
                                        seq_idx, last_position, :
                                    ]
                                else:
                                    # If the token ID is not found, you might want to handle this case.
                                    # For example, use the embeddings of the last token of the sequence
                                    # or set to zeros, depending on your application's requirements.
                                    # Here, we use the embeddings of the last token as a fallback.
                                    selected_embs[seq_idx, :] = out[seq_idx, -1, :]
                                    self.logger.log(
                                        f"EOS token not found for sequence {i}"
                                    )

                            # Update the embeddings tensor with the selected embeddings
                            embs[i : i + current_batch_size, :] = selected_embs
                        elif utils.convert_to_number(reduction) is not None:
                            # Select the embeddings for the i token of each sequence in the batch
                            embs[i : i + current_batch_size, :] = out[
                                :, utils.convert_to_number(reduction), :
                            ]
                        else:
                            raise ValueError("Unsupported reduction option")
                        del out
                        i = i + current_batch_size
                        if log_interval != -1 and i % log_interval == 0:
                            self.logger.log(
                                f" {i} / {len(seq_dataset)} | {time.time() - start:.2f}s "
                            )
            mem_usage = utils.print_gpu_utilization(memory_usage, device)
            if mem_usage > max_mem_usage:
                max_mem_usage = mem_usage
            extraction_time = time.time() - start_extraction_time
            torch.save(embs, f"{self.logger.base_dir}/{self.logger.experiment_name}.pt")
            file_size_bytes = os.path.getsize(
                f"{self.logger.base_dir}/{self.logger.experiment_name}.pt"
            )
            file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
            report = {
                "encoding_time_needed": f"{enc_time:.4f}",
                "extraction_time_needed": f"{extraction_time:.4f}",
                "avg_time_per_seq": f"{extraction_time / len(seq_dataset):.4f}",
                "dataset_len": len(seq_dataset),
                "embeddings_dim": embs.shape,
                "data_type": data_type,
                "layer": layer,
                "reduction": reduction,
                "batch_size": batch_size,
                "embeddings_file_size_mb": f"{file_size_mb:.2f}",
                "max_vram_usage_mb": max_mem_usage,
            }
            self.logger.save_data(report, "report")
            t = torch.load(f"{self.logger.base_dir}/{self.logger.experiment_name}.pt")
            self.logger.log(
                f'Saved embeddings ({t.shape[1]}-d) as "{self.logger.experiment_name}.pt" ({time.time() - start_enc_time:.2f}s)'
            )

            if torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(
                    f"{self.logger.base_dir}/memory_profiler.pickle"
                )
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(enabled=None)
            return
        except:
            if torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(
                    f"{self.logger.base_dir}/memory_profiler.pickle"
                )
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(enabled=None)
            stack_trace = traceback.format_exc()
            self.logger.log(stack_trace)

    def evaluate(self, data_type):
        pass

    def categorical_encode(self, data, max_length="default"):
        encs = utils.categorical_encode(
            data["aa_seq"].values,
            self.tokenizer,
            max(data["len"].values) if max_length == "default" else max_length,
            logger=self.logger,
            model_name="esm2",
        )
        return encs


class ESMCFamily(IPretrainedProteinLanguageModel):

    def __init__(self, esm_version: str, logger: l.Logger, task: str = "regression"):
        super().__init__(logger, task)
        self.version = esm_version
        if self.task == "masked_lm":
            raise ValueError("Masked LM not supported for ESM Cambrian")
        elif self.task == "token_classification":
            self.py_model = PlmfitEsmCForSequenceClassification.from_pretrained(
                f"{esm_version}"
            )
        elif self.task == "extract_embeddings":
            self.py_model = PlmfitEsmCForEmbdeddingsExtraction.from_pretrained(
                f"{esm_version}"
            )
        else:
            self.py_model = PlmfitEsmCForSequenceClassification.from_pretrained(
                f"{esm_version}"
            )
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.transformer.blocks)
        self.emb_layers_dim = self.py_model.embed.embedding_dim
        self.tokenizer = self.py_model.tokenizer
        self.layer_to_use = self.no_layers - 1
        self.experimenting = False

    def categorical_encode(self, data, max_length="default"):
        encs = utils.categorical_encode(
            data["aa_seq"].values,
            self.tokenizer,
            max(data["len"].values) if max_length == "default" else max_length,
            logger=self.logger,
            model_name="esmc",
        )
        return encs


class ProteinBERTFamily(IPretrainedProteinLanguageModel):
    tokenizer: Tokenizer

    def __init__(self, logger=None, task="regression"):
        super().__init__(logger, task)
        self.name = "bert-base"
        if self.task == "masked_lm":
            self.py_model: ProteinBertForMaskedLM = (
                ProteinBertForMaskedLM.from_pretrained("bert-base")
            )
            self.output_dim = self.py_model.mlm.vocab_size
        elif self.task == "token_classification":
            self.py_model: ProteinBertForTokenClassification = (
                ProteinBertForTokenClassification.from_pretrained("bert-base")
            )
            self.output_dim = self.py_model.classifier.out_features
        elif self.task == "extract_embeddings":
            self.py_model = ProteinBertForEmbeddingsExtraction.from_pretrained(
                "bert-base"
            )
        else:
            self.py_model: ProteinBertForSequenceClassification = (
                ProteinBertForSequenceClassification.from_pretrained("bert-base")
            )
            self.output_dim = self.py_model.classifier.out_features
        self.no_parameters = utils.get_parameters(self.py_model)
        self.no_layers = len(self.py_model.bert.encoder.layer)
        self.emb_layers_dim = self.py_model.bert.encoder.layer[
            0
        ].attention.output.dense.out_features
        self.tokenizer = utils.load_tokenizer(self.name)
        self.layer_to_use = self.no_layers - 1
        self.experimenting = False
        self.config = self.py_model.config

    def categorical_encode(self, data, max_length="default"):
        encs = utils.categorical_encode(
            data["aa_seq"].values,
            self.tokenizer,
            max(data["len"].values) if max_length == "default" else max_length,
            add_bos=True,
            add_eos=True,
            logger=self.logger,
            model_name=self.name,
        )
        return encs

    def extract_embeddings(
        self, data_type, batch_size=1, layer=11, reduction="mean", log_interval=1000
    ):
        try:
            self.set_layer_to_use(layer)
            layer = (
                self.layer_to_use + 1
            )  # Adjusted for considering input embeddings as well
            self.logger.log(f"Extracting embeddings from layer: {layer}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            memory_usage = psutil.virtual_memory()
            max_mem_usage = utils.print_gpu_utilization(memory_usage, device)
            fp16 = False
            device_ids = []
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(1 / 1)
                torch.cuda.memory._record_memory_history(
                    enabled="all", max_entries=100000
                )
                device = "cuda:0"
                fp16 = True
                self.logger.log(f"Available GPUs : {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.logger.log(
                        f"Running on {torch.cuda.get_device_properties(i).name}"
                    )
                    device_ids.append(i)

            else:
                self.logger.log(f"No gpu found rolling device back to {device}")

            data = utils.load_dataset(data_type)
            # data = data.sample(10)
            start_enc_time = time.time()
            self.logger.log(f"Encoding {data.shape[0]} sequences....")
            encs = self.categorical_encode(data)
            mem_usage = utils.print_gpu_utilization(memory_usage, device)
            if mem_usage > max_mem_usage:
                max_mem_usage = mem_usage
            self.logger.log(f"Encoding completed! {time.time() -  start_enc_time:.4f}s")
            enc_time = time.time() - start_enc_time
            encs = encs.to(device)
            seq_dataset = data_utils.TensorDataset(encs)
            seq_loader = data_utils.DataLoader(
                seq_dataset, batch_size=batch_size, shuffle=False
            )
            self.logger.log(
                f"Extracting embeddings for {len(seq_dataset)} sequences..."
            )

            # FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well
            embs = torch.zeros((len(seq_dataset), self.emb_layers_dim)).to(device)
            self.py_model = self.py_model.to(device)
            i = 0
            self.py_model.eval()
            start_extraction_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16, cache_enabled=False):
                    for batch in seq_loader:
                        start = time.time()
                        current_batch_size = batch[0].shape[0]
                        if layer == "logits":
                            out = self.py_model(batch[0]).logits
                        else:
                            model_output = self.py_model(batch[0])
                            hidden_states = model_output.hidden_states

                            # Log the shape of each layer's embeddings for the first batch if we are unsure what our model outputs
                            # if i == 0:
                            #     for layer_index, layer_output in enumerate(hidden_states):
                            #         self.logger.log(f'Layer {layer_index} shape: {layer_output.shape}')

                            # Now select the specific layer's output
                            out = hidden_states[layer]
                        if reduction == "mean":
                            embs[i : i + current_batch_size, :] = torch.mean(out, dim=1)
                            if i == 0:
                                self.logger.log(f"{(torch.mean(out, dim=1)).size()}")
                        elif reduction == "sum":
                            embs[i : i + current_batch_size, :] = torch.sum(out, dim=1)
                        elif reduction == "bos":
                            # Select the embeddings for the first token of each sequence in the batch
                            embs[i : i + current_batch_size, :] = out[:, 0, :]
                        elif reduction == "eos":
                            # Initialize a tensor to store the selected embeddings
                            selected_embs = torch.zeros(
                                current_batch_size, out.shape[2], device=out.device
                            )

                            # Iterate over each sequence in the batch
                            for seq_idx in range(current_batch_size):
                                # Find the positions of the token with ID equal to 2 in the current sequence
                                token_positions = (batch[0][seq_idx] == 2).nonzero(
                                    as_tuple=True
                                )[0]

                                # Check if the token ID is present in the sequence
                                if len(token_positions) > 0:
                                    # Select the position of the last occurrence of the token
                                    last_position = token_positions[-1].item()

                                    # Select the embeddings for the last occurrence of the token
                                    selected_embs[seq_idx, :] = out[
                                        seq_idx, last_position, :
                                    ]
                                else:
                                    # If the token ID is not found, you might want to handle this case.
                                    # For example, use the embeddings of the last token of the sequence
                                    # or set to zeros, depending on your application's requirements.
                                    # Here, we use the embeddings of the last token as a fallback.
                                    selected_embs[seq_idx, :] = out[seq_idx, -1, :]
                                    self.logger.log(
                                        f"EOS token not found for sequence {i}"
                                    )

                            # Update the embeddings tensor with the selected embeddings
                            embs[i : i + current_batch_size, :] = selected_embs
                        elif utils.convert_to_number(reduction) is not None:
                            # Select the embeddings for the i token of each sequence in the batch
                            embs[i : i + current_batch_size, :] = out[
                                :, utils.convert_to_number(reduction), :
                            ]
                        else:
                            raise ValueError("Unsupported reduction option")
                        del out
                        i = i + current_batch_size
                        if log_interval != -1 and i % log_interval == 0:
                            self.logger.log(
                                f" {i} / {len(seq_dataset)} | {time.time() - start:.2f}s "
                            )
            mem_usage = utils.print_gpu_utilization(memory_usage, device)
            if mem_usage > max_mem_usage:
                max_mem_usage = mem_usage
            extraction_time = time.time() - start_extraction_time
            torch.save(embs, f"{self.logger.base_dir}/{self.logger.experiment_name}.pt")
            file_size_bytes = os.path.getsize(
                f"{self.logger.base_dir}/{self.logger.experiment_name}.pt"
            )
            file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes
            report = {
                "encoding_time_needed": f"{enc_time:.4f}",
                "extraction_time_needed": f"{extraction_time:.4f}",
                "avg_time_per_seq": f"{extraction_time / len(seq_dataset):.4f}",
                "dataset_len": len(seq_dataset),
                "embeddings_dim": embs.shape,
                "data_type": data_type,
                "layer": layer,
                "reduction": reduction,
                "batch_size": batch_size,
                "embeddings_file_size_mb": f"{file_size_mb:.2f}",
                "max_vram_usage_mb": max_mem_usage,
            }
            self.logger.save_data(report, "report")
            t = torch.load(f"{self.logger.base_dir}/{self.logger.experiment_name}.pt")
            self.logger.log(
                f'Saved embeddings ({t.shape[1]}-d) as "{self.logger.experiment_name}.pt" ({time.time() - start_enc_time:.2f}s)'
            )

            if torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(
                    f"{self.logger.base_dir}/memory_profiler.pickle"
                )
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(enabled=None)
            return
        except:
            if torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(
                    f"{self.logger.base_dir}/memory_profiler.pickle"
                )
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(enabled=None)
            stack_trace = traceback.format_exc()
            self.logger.log(stack_trace)

    def fine_tune(self, data_type, fine_tuner, train_split_name, optimizer, loss_f):
        # Implementation for BERT...
        pass


# TODO: Implement AnkhFamily in modern PLMFit
class AnkhFamily(IPretrainedProteinLanguageModel):

    tokenizer: AutoTokenizer

    def __init__(self, ankh_version: str):
        super().__init__()
        self.version = ankh_version
        self.py_model = AutoModel.from_pretrained(
            f"ElnaggarLab/{ankh_version}", output_hidden_states=True
        )
        self.no_parameters = utils.get_parameters(self.py_model.encoder)
        self.no_layers = self.py_model.config.num_layers
        self.output_dim = self.py_model.config.vocab_size
        self.emb_layers_dim = self.py_model.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(f"ElnaggarLab/{ankh_version}")

    def extract_embeddings(
        self, data_type, batch_size=4, layer=48, reduction="mean", mut_pos=None
    ):
        logger = l.Logger(
            f"logger_extract_embeddings_{data_type}_{self.version}_layer{layer}_{reduction}"
        )
        device = "cpu"
        fp16 = False
        device_ids = []
        if torch.cuda.is_available():
            device = "cuda:0"
            fp16 = True
            logger.log(f"Available GPUs : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.log(f" Running on {torch.cuda.get_device_properties(i).name}")
                device_ids.append(i)

        else:
            logger.log(f" No gpu found rolling device back to {device}")
        data = utils.load_dataset(data_type)
        start_enc_time = time.time()
        logger.log(f" Encoding {data.shape[0]} sequences....")
        encs = self.categorical_encode(
            data["aa_seq"].values, self.tokenizer, max(data["len"].values)
        )
        logger.log(f" Encoding completed! {time.time() -  start_enc_time:.4f}s")
        encs = encs.to(device)

        seq_dataset = data_utils.TensorDataset(encs)
        seq_loader = data_utils.DataLoader(
            seq_dataset, batch_size=batch_size, shuffle=False
        )
        logger.log(f"Extracting embeddings for {len(seq_dataset)} sequences...")

        if type(layer) == int:
            layer = [layer]

        if type(reduction) == str:
            reduction = [reduction]

        embs = torch.zeros(
            len(layer), len(reduction), len(seq_dataset), self.emb_layers_dim
        ).to(
            device
        )  ### FIX: Find embeddings dimension either hard coded for model or real the pytorch model of ProGen. Maybe add reduction dimension as well

        self.py_model = self.py_model.encoder.to(device)
        self.py_model.eval()

        i = 0
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled= fp16):
            for batch in seq_loader:
                start = time.time()
                out = self.py_model(batch[0]).hidden_states
                for j in range(len(layer)):
                    lay = layer[j]
                    for k in range(len(reduction)):
                        if reduction[k] == "mean":
                            embs[j, k, i : i + batch_size, :] = torch.mean(
                                out[lay], dim=1
                            )
                        elif reduction[k] == "sum":
                            embs[j, k, i : i + batch_size, :] = torch.sum(
                                out[lay], dim=1
                            )
                        elif reduction[k] == "eos":
                            embs[j, k, i : i + batch_size, :] = out[lay][:, -1]
                        elif reduction[k].startswith("pos"):
                            embs[j, k, i : i + batch_size, :] = out[lay][
                                :, int(reduction[k][3:])
                            ]
                        elif reduction[k] == "mut_mean":
                            n_pos = len(mut_pos)
                            f_pos = mut_pos[0]
                            for pos in mut_pos[1:]:
                                out[lay][:, f_pos] = torch.add(
                                    out[lay][:, f_pos], out[lay][:, pos]
                                )
                            embs[j, k, i : i + batch_size, :] = torch.div(
                                out[lay][:, f_pos], n_pos
                            )
                        else:
                            raise "Unsupported reduction option"
                del out
                i = i + batch_size
                logger.log(
                    f" {i} / {len(seq_dataset)} | {time.time() - start:.2f}s "
                )  # | memory usage : {100 - memory_usage.percent:.2f}%

        os.makedirs(f"./plmfit/data/{data_type}/embeddings", exist_ok=True)
        for j in range(len(layer)):
            lay = layer[j]
            for k in range(len(reduction)):
                tmp = embs[j, k].detach().clone()
                torch.save(
                    tmp,
                    f"./plmfit/data/{data_type}/embeddings/{data_type}_{self.version}_embs_layer{layer[j]}_{reduction[k]}.pt",
                )
                logger.log(
                    f'Saved embeddings ({tmp.shape[1]}-d) as "{data_type}_{self.version}_embs_layer{layer[j]}_{reduction[k]}.pt" ({time.time() - start_enc_time:.2f}s)'
                )
                del tmp
        return

    def categorical_encode(self, seqs, tokenizer, max_len):
        seq_tokens = tokenizer.get_vocab()["<pad>"] * torch.ones(
            (len(seqs), max_len + 1), dtype=int
        )  ### Adding  to max_len because ESMTokenizer adds cls and eos tokens in the begging and the neding of aa_seq
        for itr, seq in enumerate(seqs):
            tok_seq = torch.tensor(tokenizer.encode(seq))
            seq_tokens[itr][: tok_seq.shape[0]] = tok_seq
        return seq_tokens


class SapiesFamily:
    pass
