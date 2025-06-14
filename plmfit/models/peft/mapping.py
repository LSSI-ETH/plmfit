from plmfit.models.peft.peft_model import PeftModel

def get_peft_model(model, peft_config):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """

    model_config = model.config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    if peft_config.peft_type == "LORA":
        peft_config = _prepare_lora_config(peft_config, model_config)
        return PeftModel(model, peft_config)
    elif peft_config.peft_type == "BOTTLENECK":
        peft_config = _prepare_bottleneck_config(peft_config, model_config)
        return PeftModel(model, peft_config)

TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "progen": ["qkv_proj"],
    "esm": ["query", "key", "value"],
    "bert": ["query", "key", "value"],
    "linear": ["query", "key", "value"],
    "esmc": ["layernorm_qkv.1"],
}

# TODO adapt this to our plms
TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING = {
    "progen": ["mlp"],
    "esm2": r".*encoder\.layer\.[^.]*\.output$",
    "bert": ["output"],
    "esmc": ["layernorm_qkv.1"],
}

def _prepare_lora_config(peft_config, model_config):
    if peft_config.target_modules is None:
        if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
    return peft_config

def _prepare_bottleneck_config(peft_config, model_config):
    if peft_config.target_modules is None:
        if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        peft_config.target_modules = TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING[model_config["model_type"]]
        # This is the only way to target only specific layers with ESM having two modules named 'ouput' in each layer, 
        # since we have to use regexp to get the desired module but this then means layers_to_transform does not work
        if peft_config.layers_to_transform is not None and model_config["model_type"] == 'esm': 
            peft_config.target_modules = f'.*encoder\\.layer\\.{peft_config.layers_to_transform}\\.output$'
    return peft_config
