from peft import PeftModel as HfPeftModel
from peft import PeftConfig
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from plmfit.models.peft.tuners.bottleneck_adapters import BottleneckModel

class PeftModel(HfPeftModel):
    def __init__(self, model, peft_config: PeftConfig):
        PEFT_TYPE_TO_MODEL_MAPPING["BOTTLENECK"] = BottleneckModel
        super().__init__(model, peft_config)