import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)

try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        try:
            from deepspeed import zero
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        except ImportError:
            return param.detach().cpu().clone()
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def _forward_candidate_llm(self, base_model, inputs_embeds, attention_mask, position_ids):
        llm = base_model.get_model()
        if getattr(getattr(llm, "config", None), "_attn_implementation", None) != "sdpa":
            return llm(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state

        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        if attention_mask is None:
            key_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        else:
            key_mask = attention_mask.to(device=device, dtype=torch.bool)
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len) & key_mask.view(batch_size, 1, 1, seq_len)

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        cache_position = torch.arange(0, seq_len, dtype=torch.long, device=device)

        hidden_states = inputs_embeds
        for decoder_layer in llm.layers:
            if getattr(llm, "gradient_checkpointing", False) and llm.training:
                layer_outputs = llm._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,
                    False,
                    False,
                    cache_position,
                )
            else:
                _layer_kwargs = dict(
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                # cache_position is supported only in some transformers versions; skip it otherwise.
                import inspect as _inspect
                if "cache_position" in _inspect.signature(decoder_layer.forward).parameters:
                    _layer_kwargs["cache_position"] = cache_position
                layer_outputs = decoder_layer(hidden_states, **_layer_kwargs)
            hidden_states = layer_outputs[0]
        hidden_states = llm.norm(hidden_states)
        return hidden_states

    def _compute_candidate_scores(
        self,
        model,
        candidate_input_ids,
        candidate_attention_mask,
        candidate_labels,
        candidate_graph_indices,
        graph_inputs,
    ):
        base_model = model.module if hasattr(model, "module") else model
        flat_scores = []
        candidate_batch_size = int(getattr(self.args, "graph_candidate_batch_size", 0) or 0)
        if candidate_batch_size <= 0:
            candidate_batch_size = int(candidate_input_ids.shape[0])

        for start in range(0, int(candidate_input_ids.shape[0]), candidate_batch_size):
            end = min(start + candidate_batch_size, int(candidate_input_ids.shape[0]))
            sample_indices = candidate_graph_indices[start:end]
            sub_graph_inputs = {
                key: value.index_select(0, sample_indices)
                for key, value in graph_inputs.items()
                if value is not None
            }
            (
                _,
                position_ids,
                mm_attention_mask,
                _,
                inputs_embeds,
                mm_labels,
                _,
            ) = base_model.prepare_inputs_labels_for_multimodal(
                candidate_input_ids[start:end],
                None,
                candidate_attention_mask[start:end],
                None,
                candidate_labels[start:end],
                None,
                return_graph_meta=True,
                **sub_graph_inputs,
            )
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0)
            last_hidden_state = self._forward_candidate_llm(
                base_model,
                inputs_embeds,
                mm_attention_mask,
                position_ids,
            )
            last_hidden_state = torch.nan_to_num(last_hidden_state, nan=0.0, posinf=0.0, neginf=0.0).clamp(-16.0, 16.0)
            shift_hidden = last_hidden_state[:, :-1]
            shift_labels = mm_labels[:, 1:]
            label_mask = shift_labels.ne(-100)
            scores = torch.zeros(shift_labels.shape[0], dtype=torch.float32, device=shift_hidden.device)
            if label_mask.any():
                target_hidden = shift_hidden[label_mask]
                target_labels = shift_labels[label_mask]
                target_logits = base_model.lm_head(target_hidden)
                target_logits = torch.nan_to_num(target_logits, nan=-1e4, posinf=1e4, neginf=-1e4).clamp(-100.0, 100.0)
                target_log_probs = F.log_softmax(target_logits.float(), dim=-1)
                target_scores = target_log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
                row_ids = torch.arange(shift_labels.shape[0], device=shift_labels.device).unsqueeze(1).expand_as(shift_labels)
                scores.index_add_(0, row_ids[label_mask], target_scores)
                if getattr(self.args, "graph_candidate_score_reduction", "mean") == "mean":
                    scores = scores / label_mask.sum(dim=1).clamp_min(1)
            flat_scores.append(scores)
        return torch.cat(flat_scores, dim=0)

    def _compute_candidate_ranking_loss(self, model, inputs, candidate_inputs):
        candidate_input_ids = candidate_inputs["candidate_input_ids"]
        candidate_attention_mask = candidate_inputs["candidate_attention_mask"]
        candidate_labels = candidate_inputs["candidate_labels"]
        candidate_graph_indices = candidate_inputs["candidate_graph_indices"]
        candidate_group_sizes = candidate_inputs["candidate_group_sizes"]
        graph_inputs = {
            "graphs": inputs.get("graphs"),
            "graph_emb": inputs.get("graph_emb"),
            "graph_node_mask": inputs.get("graph_node_mask"),
            "graph_hops": inputs.get("graph_hops"),
            "graph_ppr_scores": inputs.get("graph_ppr_scores"),
            "graph_role_ids": inputs.get("graph_role_ids"),
            "graph_role_mask": inputs.get("graph_role_mask"),
            "graph_text_mask": inputs.get("graph_text_mask"),
            "graph_image_mask": inputs.get("graph_image_mask"),
            "graph_domain_id": inputs.get("graph_domain_id"),
        }
        scores = self._compute_candidate_scores(
            model,
            candidate_input_ids,
            candidate_attention_mask,
            candidate_labels,
            candidate_graph_indices,
            graph_inputs,
        )
        losses = []
        offset = 0
        for group_size in candidate_group_sizes.tolist():
            group_size = int(group_size)
            if group_size <= 1:
                offset += max(group_size, 0)
                continue
            group_scores = scores[offset : offset + group_size].unsqueeze(0)
            target = torch.zeros(1, dtype=torch.long, device=group_scores.device)
            losses.append(F.cross_entropy(group_scores, target))
            offset += group_size
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()

    def _center_only_graph_inputs(self, graph_inputs):
        center_graph_inputs = dict(graph_inputs)
        graph_node_mask = graph_inputs.get("graph_node_mask")
        graph_emb = graph_inputs.get("graph_emb")
        if graph_node_mask is None or graph_emb is None:
            return None
        center_mask = torch.zeros_like(graph_node_mask, dtype=torch.bool)
        if center_mask.dim() != 2 or center_mask.shape[1] < 2:
            return None
        center_mask[:, :2] = graph_node_mask[:, :2].to(dtype=torch.bool)
        center_graph_inputs["graph_node_mask"] = center_mask
        if graph_inputs.get("graph_ppr_scores") is not None:
            center_ppr_scores = torch.zeros_like(graph_inputs["graph_ppr_scores"])
            center_ppr_scores[:, :2] = 1.0
            center_graph_inputs["graph_ppr_scores"] = center_ppr_scores
        return center_graph_inputs

    def _compute_center_preserve_loss(self, model, inputs, candidate_inputs):
        weight = float(getattr(self.args, "graph_center_preserve_loss_weight", 0.0))
        if weight == 0.0:
            return None
        candidate_input_ids = candidate_inputs["candidate_input_ids"]
        candidate_attention_mask = candidate_inputs["candidate_attention_mask"]
        candidate_labels = candidate_inputs["candidate_labels"]
        candidate_graph_indices = candidate_inputs["candidate_graph_indices"]
        candidate_group_sizes = candidate_inputs["candidate_group_sizes"]
        graph_inputs = {
            "graphs": inputs.get("graphs"),
            "graph_emb": inputs.get("graph_emb"),
            "graph_node_mask": inputs.get("graph_node_mask"),
            "graph_hops": inputs.get("graph_hops"),
            "graph_ppr_scores": inputs.get("graph_ppr_scores"),
            "graph_role_ids": inputs.get("graph_role_ids"),
            "graph_role_mask": inputs.get("graph_role_mask"),
            "graph_text_mask": inputs.get("graph_text_mask"),
            "graph_image_mask": inputs.get("graph_image_mask"),
            "graph_domain_id": inputs.get("graph_domain_id"),
        }
        center_graph_inputs = self._center_only_graph_inputs(graph_inputs)
        if center_graph_inputs is None:
            return None
        enhanced_scores = self._compute_candidate_scores(
            model,
            candidate_input_ids,
            candidate_attention_mask,
            candidate_labels,
            candidate_graph_indices,
            graph_inputs,
        )
        with torch.no_grad():
            center_scores = self._compute_candidate_scores(
                model,
                candidate_input_ids,
                candidate_attention_mask,
                candidate_labels,
                candidate_graph_indices,
                center_graph_inputs,
            )
        losses = []
        offset = 0
        slack = float(getattr(self.args, "graph_center_preserve_margin", 0.0))
        min_center_margin = float(getattr(self.args, "graph_center_preserve_min_center_margin", 0.0))
        for group_size in candidate_group_sizes.tolist():
            group_size = int(group_size)
            if group_size <= 1:
                offset += max(group_size, 0)
                continue
            center_group = center_scores[offset : offset + group_size]
            enhanced_group = enhanced_scores[offset : offset + group_size]
            center_target = center_group[0]
            enhanced_target = enhanced_group[0]
            center_neg = center_group[1:].max()
            enhanced_neg = enhanced_group[1:].max()
            center_margin = center_target - center_neg
            if center_margin.detach() >= min_center_margin and center_group.argmax().detach().item() == 0:
                enhanced_margin = enhanced_target - enhanced_neg
                losses.append(F.relu(center_margin.detach() - enhanced_margin + slack))
            offset += group_size
        if not losses:
            return enhanced_scores.sum() * 0.0
        return torch.stack(losses).mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        candidate_keys = [
            "candidate_input_ids",
            "candidate_labels",
            "candidate_attention_mask",
            "candidate_graph_indices",
            "candidate_group_sizes",
        ]
        candidate_inputs = {key: inputs.pop(key, None) for key in candidate_keys}
        has_candidate_loss = (
            candidate_inputs["candidate_input_ids"] is not None
            and float(getattr(self.args, "graph_candidate_loss_weight", 0.0)) != 0.0
        )

        # Reset the accumulated domain loss at each step; the encoder accumulates it during forward passes.
        _bm = model.module if hasattr(model, "module") else model
        _enc = getattr(_bm.get_model(), "graph_soft_token_encoder", None)
        if _enc is not None and hasattr(_enc, "reset_domain_losses"):
            _enc.reset_domain_losses()

        outputs = None
        loss = None
        lm_loss_weight = float(getattr(self.args, "graph_lm_loss_weight", 1.0))
        if lm_loss_weight != 0.0:
            outputs = model(**inputs)
            lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss = lm_loss * lm_loss_weight

        if has_candidate_loss:
            candidate_loss = self._compute_candidate_ranking_loss(model, inputs, candidate_inputs)
            weighted_candidate_loss = float(getattr(self.args, "graph_candidate_loss_weight", 1.0)) * candidate_loss
            loss = weighted_candidate_loss if loss is None else loss + weighted_candidate_loss

        # --- Innovation #1 & #3: domain-adversarial losses (GRL) ---
        # Use domain logits after the final graph forward pass (candidate_loss);
        # otherwise the computation graph may already have been released.
        domain_loss_weight = float(getattr(self.args, "graph_domain_adv_loss_weight", 0.0))
        modality_domain_weight = float(getattr(self.args, "graph_modality_domain_loss_weight", 0.0))
        if domain_loss_weight != 0.0 or modality_domain_weight != 0.0:
            base_model = model.module if hasattr(model, "module") else model
            encoder = getattr(base_model.get_model(), "graph_soft_token_encoder", None)
            if domain_loss_weight != 0.0 and encoder is not None and hasattr(encoder, "domain_adversarial_loss"):
                dom_loss = encoder.domain_adversarial_loss()
                if dom_loss is not None:
                    weighted_dom_loss = domain_loss_weight * dom_loss
                    loss = weighted_dom_loss if loss is None else loss + weighted_dom_loss
            if modality_domain_weight != 0.0 and encoder is not None and hasattr(encoder, "modality_domain_loss"):
                mod_loss = encoder.modality_domain_loss()
                if mod_loss is not None:
                    weighted_mod_loss = modality_domain_weight * mod_loss
                    loss = weighted_mod_loss if loss is None else loss + weighted_mod_loss
            # End-to-end structural prior loss
            struct_prior_weight = float(getattr(self.args, "graph_structure_prior_loss_weight", 0.0))
            if struct_prior_weight != 0.0 and encoder is not None and hasattr(encoder, "structure_prior_loss"):
                struct_loss = encoder.structure_prior_loss(inputs.get("graph_label_id"))
                if struct_loss is not None:
                    loss = struct_prior_weight * struct_loss if loss is None else loss + struct_prior_weight * struct_loss
            center_preserve_loss = self._compute_center_preserve_loss(model, inputs, candidate_inputs)
            if center_preserve_loss is not None:
                weighted_center_preserve_loss = (
                    float(getattr(self.args, "graph_center_preserve_loss_weight", 0.0)) * center_preserve_loss
                )
                loss = weighted_center_preserve_loss if loss is None else loss + weighted_center_preserve_loss

        if loss is None:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if loss is not None and hasattr(loss, "item"): print(f"[step] loss={loss.item():.4f}", flush=True, file=__import__("sys").stderr)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if (
                        "mm_projector" in name
                        or "graph_projector" in name
                        or "graph_decoder" in name
                        or "graph_soft_token_encoder" in name
                        or "graph_text_decoder" in name
                        or "graph_image_decoder" in name
                        or "graph_edge_predictor" in name
                    )
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = [
                'mm_projector',
                'vision_resampler',
                'graph_projector',
                'graph_decoder',
                'graph_soft_token_encoder',
                'graph_text_decoder',
                'graph_image_decoder',
                'graph_edge_predictor',
            ]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
