import torch
from plmfit.language_models.progen2.models.progen.modeling_progen import ProGenForCausalLM

print("loading model...")
py_model = ProGenForCausalLM.from_pretrained(f'./plmfit/language_models/progen2/checkpoints/progen2-small')
print(py_model)