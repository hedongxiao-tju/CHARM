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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from model.model import *
from model.constants import DEFAULT_GRAPH_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    # Force all model layers onto a single device to avoid mixed device errors
    kwargs = {"device_map": {"": device}, **kwargs}

    if load_8bit:
        # Newer transformers versions reject passing both load_in_{4,8}bit
        # and quantization_config at the same time.
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            llm_int8_skip_modules=[
                "mm_projector",
                "graph_projector",
                "graph_soft_token_encoder",
            ],
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        is_lora_checkpoint = (
            'lora' in model_name.lower()
            or os.path.exists(os.path.join(model_path, 'adapter_config.json'))
            or os.path.exists(os.path.join(model_path, 'adapter_model.bin'))
            or os.path.exists(os.path.join(model_path, 'adapter_model.safetensors'))
        )
        if is_lora_checkpoint and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if is_lora_checkpoint and model_base is not None:
            from model.model.language_model.model_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            # Adapter checkpoints saved after 4-bit training may carry a serialized
            # quantization_config. That config should not be applied when we first
            # load the full-precision base model for LoRA merge.
            if hasattr(lora_cfg_pretrained, "quantization_config"):
                delattr(lora_cfg_pretrained, "quantization_config")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading model from base weights...')
            base_load_kwargs = dict(kwargs)
            if load_4bit or load_8bit:
                warnings.warn(
                    "Quantized loading for unmerged LoRA adapters is not supported in this code path. "
                    "Falling back to fp16 base-model loading before merging LoRA weights."
                )
                base_load_kwargs.pop("quantization_config", None)
                base_load_kwargs.pop("load_in_4bit", None)
                base_load_kwargs.pop("load_in_8bit", None)
                base_load_kwargs.pop("_load_in_4bit", None)
                base_load_kwargs.pop("_load_in_8bit", None)
                base_load_kwargs.pop("quant_method", None)
                base_load_kwargs["torch_dtype"] = torch.float16
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                **base_load_kwargs,
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional model weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            target_device = None
            target_dtype = torch.float16
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.device.type == "meta":
                    continue
                target_device = tensor.device
                if tensor.is_floating_point():
                    target_dtype = tensor.dtype
                break
            if target_device is None:
                target_device = torch.device(device if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
            non_lora_trainables = {
                k: v.to(device=target_device, dtype=target_dtype) if torch.is_floating_point(v) else v.to(device=target_device)
                for k, v in non_lora_trainables.items()
            }
            try:
                model.load_state_dict(non_lora_trainables, strict=False, assign=True)
            except TypeError:
                model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading model from base weights...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if hasattr(cfg_pretrained, "quantization_config"):
                    delattr(cfg_pretrained, "quantization_config")
                # Fix vocab_size mismatch: base model may have smaller vocab than checkpoint
                # (e.g. base=32000, checkpoint=32002 after special token additions).
                # Load base model with its own vocab_size, then resize after loading.
                _base_cfg = AutoConfig.from_pretrained(model_base)
                _base_vocab = getattr(_base_cfg, 'vocab_size', None)
                if _base_vocab is not None and getattr(cfg_pretrained, 'vocab_size', None) != _base_vocab:
                    print(f"[BUILDER] Adjusting cfg vocab_size {cfg_pretrained.vocab_size} -> {_base_vocab} to match base model", flush=True)
                    cfg_pretrained.vocab_size = _base_vocab
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            _mp_path = os.path.join(model_path, 'mm_projector.bin')
            if not os.path.exists(_mp_path):
                print(f"[BUILDER] No mm_projector.bin at {model_path}, using random init for all adapters", flush=True)
                mm_projector_weights = {}
            else:
                mm_projector_weights = torch.load(_mp_path, map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            try:
                load_result = model.load_state_dict(mm_projector_weights, strict=False, assign=True)
            except TypeError:
                load_result = model.load_state_dict(mm_projector_weights, strict=False)
                own_sd = model.state_dict()
                for k, v in mm_projector_weights.items():
                    if k in own_sd and own_sd[k].is_meta:
                        own_sd[k] = v.to(own_sd[k].dtype).to(own_sd[k].device) if hasattr(own_sd[k], 'device') else v
                model.load_state_dict(own_sd, strict=False)
            graph_keys = [k for k in mm_projector_weights if 'graph_soft_token_encoder' in k or 'graph_projector' in k]
            mm_keys = [k for k in mm_projector_weights if 'mm_projector' in k]
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(
                f"Loaded adapter weights: graph_keys={len(graph_keys)}, mm_projector_keys={len(mm_keys)}, "
                f"unexpected_keys={len(unexpected)}",
                flush=True,
            )
            # Force adapter modules to model device (CUDA) and materialize meta tensors
            _target_device = next(model.parameters()).device
            print(f"[BUILDER] target_device={_target_device}", flush=True)
            if hasattr(model.model, "mm_projector") and model.model.mm_projector is not None:
                _mp_sd = model.model.mm_projector.state_dict()
                _fixed = 0
                for k in list(_mp_sd.keys()):
                    if _mp_sd[k].is_meta:
                        for src_k, src_v in mm_projector_weights.items():
                            if src_k == f"model.mm_projector.{k}":
                                _mp_sd[k] = src_v.to(_target_device)
                                _fixed += 1
                                break
                if _fixed > 0:
                    model.model.mm_projector.load_state_dict(_mp_sd, assign=True)
                    print(f"[BUILDER] mm_projector: fixed {_fixed} meta tensors", flush=True)
                model.model.mm_projector.to(_target_device)
            if hasattr(model.model, "graph_soft_token_encoder") and model.model.graph_soft_token_encoder is not None:
                _gs_sd = model.model.graph_soft_token_encoder.state_dict()
                _fixed = 0
                for k in list(_gs_sd.keys()):
                    if _gs_sd[k].is_meta:
                        for src_k, src_v in mm_projector_weights.items():
                            if src_k == f"model.graph_soft_token_encoder.{k}":
                                _gs_sd[k] = src_v.to(_target_device)
                                _fixed += 1
                                break
                if _fixed > 0:
                    model.model.graph_soft_token_encoder.load_state_dict(_gs_sd, assign=True)
                    print(f"[BUILDER] graph_soft_token_encoder: fixed {_fixed} meta tensors", flush=True)
                model.model.graph_soft_token_encoder.to(_target_device)
            _final_dev = next(model.model.mm_projector.parameters()).device if hasattr(model.model, 'mm_projector') and model.model.mm_projector is not None else 'N/A'
            print(f"[BUILDER] FINAL mm_projector device: {_final_dev}", flush=True)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        tokenizer.add_tokens([DEFAULT_GRAPH_TOKEN], special_tokens=True)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        current_vocab_size = model.get_input_embeddings().weight.shape[0]
        target_vocab_size = len(tokenizer)
        if current_vocab_size != target_vocab_size:
            model.resize_token_embeddings(target_vocab_size)

        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model(device_map=device_map)
            if device_map != 'auto':
                vision_tower.to(device=device_map, dtype=torch.float16)
            image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
