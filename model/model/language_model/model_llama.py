#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from model.constants import DEFAULT_GRAPH_PAD_ID
from ..model_arch import LlavaMetaModel, LlavaMetaForCausalLM


@dataclass
class CausalLMOutputWithPastWithGraph(CausalLMOutputWithPast):
    feat_loss: Optional[torch.FloatTensor] = None
    graph_aux_loss: Optional[torch.FloatTensor] = None
    graph_text_rec_loss: Optional[torch.FloatTensor] = None
    graph_image_rec_loss: Optional[torch.FloatTensor] = None
    graph_edge_rec_loss: Optional[torch.FloatTensor] = None


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        graphs: Optional[torch.LongTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        graph_node_mask: Optional[torch.BoolTensor] = None,
        graph_targets: Optional[torch.FloatTensor] = None,
        graph_hops: Optional[torch.LongTensor] = None,
        graph_degrees: Optional[torch.FloatTensor] = None,
        graph_ppr_scores: Optional[torch.FloatTensor] = None,
        graph_role_ids: Optional[torch.LongTensor] = None,
        graph_role_mask: Optional[torch.BoolTensor] = None,
        graph_text_mask: Optional[torch.BoolTensor] = None,
        graph_image_mask: Optional[torch.BoolTensor] = None,
        graph_domain_id: Optional[torch.LongTensor] = None,
        graph_edges: Optional[List[torch.LongTensor]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if graph_node_mask is None and graphs is not None:
            graph_node_mask = graphs.ne(DEFAULT_GRAPH_PAD_ID)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                graphs=graphs,
                graph_emb=graph_emb,
                graph_node_mask=graph_node_mask,
                graph_hops=graph_hops,
                graph_degrees=graph_degrees,
                graph_ppr_scores=graph_ppr_scores,
                graph_role_ids=graph_role_ids,
                graph_role_mask=graph_role_mask,
                graph_text_mask=graph_text_mask,
                graph_image_mask=graph_image_mask,
                graph_domain_id=graph_domain_id,
                return_graph_meta=True,
            )

        _graph_label_id = kwargs.pop("graph_label_id", None)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        feat_loss = None
        graph_aux_loss = None
        graph_text_rec_loss = None
        graph_image_rec_loss = None
        graph_edge_rec_loss = None

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return ((outputs.loss,) + output) if outputs.loss is not None else output

        return CausalLMOutputWithPastWithGraph(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            feat_loss=feat_loss,
            graph_aux_loss=graph_aux_loss,
            graph_text_rec_loss=graph_text_rec_loss,
            graph_image_rec_loss=graph_image_rec_loss,
            graph_edge_rec_loss=graph_edge_rec_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        graphs: Optional[torch.LongTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        graph_node_mask: Optional[torch.BoolTensor] = None,
        graph_hops: Optional[torch.LongTensor] = None,
        graph_degrees: Optional[torch.FloatTensor] = None,
        graph_ppr_scores: Optional[torch.FloatTensor] = None,
        graph_role_ids: Optional[torch.LongTensor] = None,
        graph_role_mask: Optional[torch.BoolTensor] = None,
        graph_text_mask: Optional[torch.BoolTensor] = None,
        graph_image_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or graph_emb is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                graphs=graphs,
                graph_emb=graph_emb,
                graph_node_mask=graph_node_mask,
                graph_hops=graph_hops,
                graph_degrees=graph_degrees,
                graph_ppr_scores=graph_ppr_scores,
                graph_role_ids=graph_role_ids,
                graph_role_mask=graph_role_mask,
                graph_text_mask=graph_text_mask,
                graph_image_mask=graph_image_mask,
                return_graph_meta=True,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        graphs = kwargs.pop("graphs", None)
        graph_emb = kwargs.pop("graph_emb", None)
        graph_node_mask = kwargs.pop("graph_node_mask", None)
        graph_hops = kwargs.pop("graph_hops", None)
        graph_degrees = kwargs.pop("graph_degrees", None)
        graph_ppr_scores = kwargs.pop("graph_ppr_scores", None)
        graph_role_ids = kwargs.pop("graph_role_ids", None)
        graph_role_mask = kwargs.pop("graph_role_mask", None)
        graph_text_mask = kwargs.pop("graph_text_mask", None)
        graph_image_mask = kwargs.pop("graph_image_mask", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if graphs is not None:
            inputs["graphs"] = graphs
        if graph_emb is not None:
            inputs["graph_emb"] = graph_emb
        if graph_node_mask is not None:
            inputs["graph_node_mask"] = graph_node_mask
        if graph_hops is not None:
            inputs["graph_hops"] = graph_hops
        if graph_degrees is not None:
            inputs["graph_degrees"] = graph_degrees
        if graph_ppr_scores is not None:
            inputs["graph_ppr_scores"] = graph_ppr_scores
        if graph_role_ids is not None:
            inputs["graph_role_ids"] = graph_role_ids
        if graph_role_mask is not None:
            inputs["graph_role_mask"] = graph_role_mask
        if graph_text_mask is not None:
            inputs["graph_text_mask"] = graph_text_mask
        if graph_image_mask is not None:
            inputs["graph_image_mask"] = graph_image_mask
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
