from peft import LoraConfig, TaskType
from plmfit.language_models.progen2.models.progen.modeling_progen import ProGenForSequenceClassification, ProGenForCausalLM
from plmfit.language_models.progen2.models.progen.configuration_progen import ProGenConfig

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
)

from transformers import AutoModel, BertForSequenceClassification, AutoConfig, AutoModelForCausalLM


progen2 =  ProGenForSequenceClassification.from_pretrained(f'./plmfit/language_models/progen2/checkpoints/progen2-small')

print(progen2)