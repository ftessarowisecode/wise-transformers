# coding=utf-8
# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
 PyTorch DistilBERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
 part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""


import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import get_activation
from ...configuration_utils import PretrainedConfig
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    SequenceClassifierMultiLevelOutput
)
from ...models.distilbert import DistilBertModel, DistilBertPreTrainedModel
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_distilbert import DistilBertConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # See all DistilBERT models at https://huggingface.co/models?filter=distilbert
]



# @add_start_docstrings(
#     """
#     DistilBert Model transformer with a multilevel classification heads on top (more linear layers on top of the
#     pooled output) e.g. for GLUE tasks.
#     """,
#     DISTILBERT_START_DOCSTRING,
# )
class DistilBertForMultiLabelClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        
        # Define separate classifiers for each level of the taxonomy
        self.level_classifiers = nn.ModuleList([
            nn.Linear(self.distilbert.config.hidden_size, num_classes)
            for num_classes in config.num_classes_per_level
        ])
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None, 
        labels: Optional[list[list[torch.LongTensor]]] = None, # labels should be a tuple for each level (L1, L2, ... , Ln) DNW
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierMultiLevelOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        # logits = self.classifier(pooled_output)  # (bs, num_labels)
        logits_list = []
        for classifier in self.level_classifiers:
            # Apply classifier to the output of DistilBERT
            logits = classifier(pooled_output)  # Linear layer for each level
            logits_list.append(logits)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                logger.warning("!!!! config.problem_type Not setted !!!")
                # if self.num_labels == 1:
                #     self.config.problem_type = "regression"
                # elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                #     self.config.problem_type = "single_label_classification"
                # else:
                #     self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_taxonomy_classification":
                # logger.info("ENTERED IN CROSS MULTIPLE ENTROPY")
                criterions = [CrossEntropyLoss() for i in enumerate(self.level_classifiers)]  # Replace with appropriate criteria for each level
                # sum up all cross-entropy losses wrt all output layers
                # convert format of labels [[l1,l2], [l1,l2] ... ] -> [[l1, l1, ...], [l2, l2, ...]]
                labels = [torch.tensor(x, dtype=torch.long).to(self.device) for x in zip(*labels)]
                # logger.info(f"labels transposed: {labels}")

                # concatenate all labels into a single touple
                # labels = [labels_l1, labels_l2]
                loss = sum(criterions[i](logits_list[i], labels[i])
                               for i, _, _ in zip(range(len(logits_list)), criterions, labels)).to(self.device)


        if not return_dict:
            output = (logits for logits in logits_list) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierMultiLevelOutput(
            loss=loss,
            logits=logits_list,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
