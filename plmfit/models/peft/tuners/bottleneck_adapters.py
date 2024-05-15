import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import PeftConfig
from transformers.activations import ACT2FN

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists, ModulesToSaveWrapper

@dataclass
class BottleneckConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Bottleneck`].

    Args:
        bottleneck_size (`int`): The size of the bottleneck.
        non_linearity (`str`): The non-linearity to apply to the bottleneck.
        adapter_dropout (`float`, optional): The dropout probability of the bottleneck. Default to 0.0
        bias ('str'): Bias type for Bottleneck. Can be 'none', 'all' or 'adapter_only'. Default to 'none'.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can be either a
            constant factor (float) or the string "learned", in which case the scaling factor is learned. Defaults to
            1.0.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Adapter to.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter".
        modules_to_save (`List[str]`):List of modules apart from Bottleneck adapter layers to be set as trainable
            and saved in the final checkpoint.
    """
    bottleneck_size : int = field(default=256, metadata={"help": "The size of the bottleneck"})
    non_linearity : str = field(default="tanh", metadata={"help": "The non-linearity to apply to the bottleneck"})
    adapter_dropout : float = field(default=0.0, metadata={"help": "The dropout probability of the bottleneck, default to 0.0"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Adapter."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    scaling: Union[float, str] = 1.0
    bias: str = field(default="none", metadata={"help": "Bias type for Bottleneck. Can be 'none', 'all' or 'adapter_only'"})
    init_weights: str = field(default="bert", metadata={"help": "Initialization method for the weights of the adapter modules."})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Adapter layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )

    def __post_init__(self):
        self.peft_type = 'BOTTLENECK'


class BottleneckModel(BaseTuner):
    """
    Creates Bottleneck adapter model for a pretrained trainsformers model.

    Args:
        model ('transformers.PreTrainedModel'): The pretrained model to be adapted.
        config (`BottleneckConfig`): The configuration of the Bottleneck adapter.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
    
    Returns:
        `torch.nn.Module`: The Bottleneck adapter model.
    
    Example::

        >>> from transformers import AutoModelForCausalLM, BottleneckConfig
        >>> from peft import BottleneckModel, BottleneckConfig
        >>> config = BottleneckConfig(
            peft_type="BOTTLNECK", task="CAUSAL_LM", target_modules=["gate_proj", "up_proj", "down_proj"],
            bottleneck_size=256, non_linearity="tanh",
        )
        >>> model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf") 
        >>> bottleneck_model = BottleneckModel(config, model)

    **Attribute**:
        - **model** (`transformers.PreTrainedModel`): The pretrained model to be adapted.
        - **peft_config** (`BottleneckConfig`): The configuration of the Bottleneck adapter.
    """

    prefix: str = "adapter_"

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _create_and_replace(
        self,
        bottleneck_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        kwargs = {
            "bottleneck_size": bottleneck_config.bottleneck_size,
            "non_linearity": bottleneck_config.non_linearity,
            "adapter_dropout": bottleneck_config.adapter_dropout,
            "scaling": bottleneck_config.scaling,
            "init_weights": bottleneck_config.init_weights,
            "bias": bottleneck_config.bias
        }

        adapter_type = "output_adapter"
        kwargs.update({"adapter_type": adapter_type})
            
        out_features = get_last_linear_out_features(target)
        new_module = AdapterLayer(out_features, out_features, **kwargs)
        
        # Create a sequential container that first executes the original target module,
        # then passes the output through the new adapter module
        augmented_module = AdapterSequential()
        augmented_module.add_module('original', target)
        augmented_module.add_module('adapter', new_module)

        # Replace the original target in the parent module with the new augmented module
        self._replace_module(parent, target_name, augmented_module)

    def _replace_module(self, parent_module, child_name, new_module):
        setattr(parent_module, child_name, new_module)
        # Ensure the new module is dispatched to the correct device
        new_module.to(next(parent_module.parameters()).device)
        
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (AdapterLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "adapter_only":
                for m in model.modules():
                    if isinstance(m, AdapterLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _prepare_adapter_config(model, config, adapter_name):
        return config

# Below code is based on https://github.com/adapter-hub/adapter-transformers/blob/master/src/transformers/adapters/modeling.py and lora.py from huggingfance PEFT
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------



class AdapterLayer(nn.Module):
    """
    Bottleneck adapter in a dense layer. The adapter can be applied after the multi-head attention layer and/or
    after the feed-forward layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        adapter_type: str,
        bottleneck_size: int,
        non_linearity: str,
        adapter_dropout: float,
        scaling: Union[float, str],
        init_weights: str,
        **kwargs,
    ):        
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.non_linearity = non_linearity
        self.scaling = scaling
        self.disable_adapters = False
        self.init_weights = init_weights
        self.adapter_type = adapter_type
        if isinstance(scaling, float):
            self.adapter_scaling = scaling
        elif scaling == "learned":
            self.adapter_scaling = nn.Parameter(torch.ones(1))


        # Actual trainable parameters
        self.adapter_norm = nn.LayerNorm(out_features)
        self.adapter_down = nn.Linear(in_features, bottleneck_size, **kwargs)
        self.act_fn = ACT2FN[self.non_linearity]
        self.adapter_up = nn.Linear(bottleneck_size, out_features, **kwargs)
        self.dropout = nn.Dropout(adapter_dropout)


    def train(self, mode: bool = True):
        self.adapter_down.train(mode)
        self.adapter_up.train(mode)

    def eval(self):
        self.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            return x
        else:
            # Normalize input features
            normalized_output = self.adapter_norm(x)
            # Apply down projection
            down_projected = self.adapter_down(normalized_output)
            # Non-linear activation
            activated = self.act_fn(down_projected)
            # Up projection
            up_projected = self.adapter_up(activated)
            # Apply dropout
            dropped_out = self.dropout(up_projected)
            # Add scaled output to original input
            output = x + self.adapter_scaling * dropped_out
            return output                

                        
def get_last_linear_out_features(model):
        # Recursively find the last nn.Linear module
        def find_last_linear(module):
            last_linear = None
            if isinstance(model, nn.Linear):
                last_linear = model
                return last_linear
            for child in module.children():
                if isinstance(child, nn.Linear):
                    last_linear = child
                elif list(child.children()):  # If the child module contains other submodules
                    potential_last = find_last_linear(child)
                    if potential_last:
                        last_linear = potential_last
            return last_linear

        last_linear = find_last_linear(model)
        if last_linear is not None:
            return last_linear.out_features
        else:
            raise ValueError("No linear layer found in the model")



class AdapterSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
        