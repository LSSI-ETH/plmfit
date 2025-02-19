from typing import Tuple, Union
import torch
from torch import nn
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)
from plmfit.shared_utils.poolers import GeneralPooler
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from transformers import EsmConfig

"""
ESM Cambrian is specifically designed for representation learning. As such, no LM class is present in the model.
"""

class PlmfitEsmCConfig(EsmConfig):
    model_type = "esmc"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def to_dict(self):
        return {**super().to_dict(), "model_type": self.model_type}


class PlmfitEsmCForSequenceClassification(ESMC):
    _keys_to_ignore_on_load_missing = ["classifier.weight", "classifier.bias"]
    _keys_to_ignore_on_load_unexpected = [
        "sequence_head.0.weight",
        "sequence_head.0.bias",
        "sequence_head.2.weight",
        "sequence_head.2.bias",
        "sequence_head.3.weight",
        "sequence_head.3.bias",
    ]
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
        use_flash_attn: bool = False,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attn,
        )

        # These are for fine_tuners to be compatible with esmc
        self.config = PlmfitEsmCConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.classifier = nn.Linear(d_model, 1)
        self.reduction = "mean"
        self.pooler = GeneralPooler()
        del self.sequence_head

    @classmethod
    def from_pretrained(
        cls, model_name, device: torch.device | None = None
    ):
        from plmfit.language_models.esmC.pretrained import load_local_model

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device, model_class=cls)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        assert isinstance(model, cls)
        return model

    def load_state_dict(self, state_dict, strict = False, assign = False):
        return super().load_state_dict(state_dict, strict, assign)

    def set_head(self, new_head):
        """
        Replace the classifier head of the model.

        Args:
            new_head (torch.nn.Module): New classifier head to replace the existing one.
        """
        self.classifier = new_head

    def trim_model(self, layer_to_use):
        self.transformer.blocks = nn.ModuleList(
            list(self.transformer.blocks.children())[: layer_to_use + 1]
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if input_ids is not None:
            input_ids = input_ids.int()
        if attention_mask is None:
            # For EMSC, a boolean mask is created in place of sequence_id (attention_mask) if not specified.
            attention_mask = input_ids != self.tokenizer.pad_token_id

        # Renaming to follow ESMC naming convention
        sequence_tokens = input_ids
        sequence_id = attention_mask

        x = self.embed(sequence_tokens)

        # B, L = x.shape[:2]

        # # If sequence_id looks like a mask.
        # if self._use_flash_attn:
        #     assert (
        #         sequence_id.dtype == torch.bool
        #     ), "sequence_id must be a boolean mask if Flash Attention is used"
        #     assert sequence_id.shape == (B, L)
        #     assert unpad_input is not None
        #     x, indices, *_ = unpad_input(x, sequence_id)  # type: ignore
        # else:
        #     indices = None

        x, _, hiddens = self.transformer(x, sequence_id=sequence_id)

        # if self._use_flash_attn:
        #     assert indices is not None
        #     assert pad_input is not None
        #     x = pad_input(x, indices, B, L)  # Back to [B, L, D]
        #     hiddens = [
        #         # Back to [[B, L, D], ...]
        #         pad_input(h, indices, B, L)
        #         for h in hiddens
        #     ]

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hiddens = torch.stack(hiddens, dim=0)  # type: ignore

        pooled_output = self.pooler(x, pooling_method=self.reduction)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=hiddens
        )

class PlmfitEsmCForEmbdeddingsExtraction(ESMC):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
        use_flash_attn: bool = False,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attn,
        )
        # These are for fine_tuners to be compatible with esmc
        self.config = PlmfitEsmCConfig(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers
        )
        self.reduction = "mean"
        self.pooler = GeneralPooler()
        del self.sequence_head

    @classmethod
    def from_pretrained(cls, model_name, device: torch.device | None = None):
        from plmfit.language_models.esmC.pretrained import load_local_model

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device, model_class=cls)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        assert isinstance(model, cls)
        return model

    def load_state_dict(self, state_dict, strict=False, assign=False):
        return super().load_state_dict(state_dict, strict, assign)

    def trim_model(self, layer_to_use):
        self.transformer.blocks = nn.ModuleList(
            list(self.transformer.blocks.children())[: layer_to_use + 1]
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if input_ids is not None:
            input_ids = input_ids.int()
        if attention_mask is None:
            # For EMSC, a boolean mask is created in place of sequence_id (attention_mask) if not specified.
            attention_mask = input_ids != self.tokenizer.pad_token_id

        # Renaming to follow ESMC naming convention
        sequence_tokens = input_ids
        sequence_id = attention_mask

        x = self.embed(sequence_tokens)

        # B, L = x.shape[:2]

        # # If sequence_id looks like a mask.
        # if self._use_flash_attn:
        #     assert (
        #         sequence_id.dtype == torch.bool
        #     ), "sequence_id must be a boolean mask if Flash Attention is used"
        #     assert sequence_id.shape == (B, L)
        #     assert unpad_input is not None
        #     x, indices, *_ = unpad_input(x, sequence_id)  # type: ignore
        # else:
        #     indices = None

        x, _, hiddens = self.transformer(x, sequence_id=sequence_id)

        # if self._use_flash_attn:
        #     assert indices is not None
        #     assert pad_input is not None
        #     x = pad_input(x, indices, B, L)  # Back to [B, L, D]
        #     hiddens = [
        #         # Back to [[B, L, D], ...]
        #         pad_input(h, indices, B, L)
        #         for h in hiddens
        #     ]

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hiddens = torch.stack(hiddens, dim=0)  # type: ignore

        pooled_output = self.pooler(x, pooling_method=self.reduction)

        return SequenceClassifierOutput(logits=pooled_output, hidden_states=hiddens)
