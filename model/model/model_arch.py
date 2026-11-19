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


from abc import ABC, abstractmethod
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from model.constants import (
    DEFAULT_GRAPH_PAD_ID,
    DEFAULT_GRAPH_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    GRAPH_TOKEN_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from model.mm_utils import get_anyres_image_grid_shape


def build_graph_projector(config):
    projector_type = getattr(config, "graph_projector_type", "linear")
    hidden_dim = getattr(config, "hidden_size")
    input_dim = getattr(config, "graph_projector_input_size", getattr(config, "graph_hidden_size"))

    if projector_type == "linear":
        return nn.Linear(input_dim, hidden_dim)

    mlp_match = re.match(r"^(\d+)-layer-mlp$", projector_type)
    if mlp_match:
        depth = int(mlp_match.group(1))
        modules = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)

    raise ValueError(f"Unknown graph projector type: {projector_type}")


class _GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward: identity. Backward: multiply incoming gradient by ``-lambda``.
    Used to make the global/transfer node representation *domain-invariant*: a
    domain classifier tries to predict which dataset a node comes from, while
    the GRL forces the upstream features to fool that classifier, removing
    dataset-specific (label-prior) signal that otherwise leaks into the
    left-out domain at zero-shot time.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradientReversal.apply(x, lambd)



class GCNPropLayer(nn.Module):
    """GCN layer with symmetric normalized Laplacian: D^{-1/2} A D^{-1/2} H W + alpha*H"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.res_gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H, edge_index, num_nodes):
        transformed = self.W(H).to(H.dtype)
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=H.device, dtype=H.dtype)
        deg.scatter_add_(0, row, torch.ones(row.shape[0], device=H.device, dtype=H.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm_val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        msg = transformed[col] * norm_val.unsqueeze(-1).to(transformed.dtype)
        agg = torch.zeros_like(transformed)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        alpha = torch.tanh(self.res_gate)
        out = agg + alpha * H
        out = self.norm(out)
        return out


class GraphSoftTokenEncoder(nn.Module):
    CENTER_TEXT_TYPE = 0
    CENTER_IMAGE_TYPE = 1
    ABSTRACT_TEXT_TYPE = 2
    CALIBRATION_TEXT_TYPE = 3
    TRANSFER_TYPE = 4
    NEIGHBOR_TYPE = 5
    NEIGHBOR_IMAGE_TYPE = 6  # Kept for loading older split-neighbor checkpoints.
    ABSTRACT_IMAGE_TYPE = 7
    CALIBRATION_IMAGE_TYPE = 8

    # Token-type ids that denote cross-dataset / transfer nodes whose
    # representation we force to be domain-invariant. NOTE: in-domain abstracts
    # (ABSTRACT_*) are intentionally excluded — they SHOULD keep domain info;
    # only TRANSFER nodes (global prototype + cross-domain evidence) are made
    # domain-invariant. The global prototype itself is tagged domain=-1 and is
    # further filtered out by the >=0 mask in _compute_domain_adv.
    GLOBAL_TYPE_IDS = (TRANSFER_TYPE,)

    def __init__(self, config):
        super().__init__()
        self.config = config
        clip_dim = int(getattr(config, "graph_hidden_size"))
        if clip_dim % 2 != 0:
            raise ValueError(f"graph_hidden_size must be even for text/image split, got {clip_dim}")
        self.modal_dim = clip_dim // 2
        hidden_dim = int(getattr(config, "hidden_size"))
        self.hidden_dim = hidden_dim
        n_heads = int(getattr(config, "graph_attn_heads", 8))
        # hidden_dim must be divisible by n_heads; fall back gracefully.
        while hidden_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.n_heads = n_heads

        # ---- learnable missing-modality placeholders (kept; cheap & useful) ----
        self.text_mask_embed = nn.Parameter(torch.zeros(self.modal_dim))
        self.image_mask_embed = nn.Parameter(torch.zeros(self.modal_dim))

        # ---- per-modality projection into the LLM hidden space ----
        self.text_projector = build_graph_projector(config)   # modal_dim -> hidden
        self.image_projector = build_graph_projector(config)

        # =====================================================================
        # Innovation #3 : Domain-Invariant Modality Fusion (DIMF)
        #   ------------------------------------------------------------------
        #   Multimodal-SPECIFIC contribution. A small network predicts, per
        #   token, a modality-fusion weight g in [0,1] (how much to trust text
        #   vs image). The fused feature is g*text + (1-g)*image plus an
        #   interaction term. CRUCIALLY, the fusion weight g is forced to be
        #   DOMAIN-INVARIANT via a gradient-reversal domain head: the model
        #   learns "which modality is reliable" WITHOUT encoding "which dataset
        #   this came from". This removes modality-preference drift across
        #   domains (e.g. Grocery relies on images, Movies on text) so the
        #   learned preference transfers to the unseen target domain.
        #   - Unlike Mario's router (which LEARNS modality preference), we
        #     REMOVE the domain-dependence of that preference (opposite goal).
        #   - This module is meaningless for text-only graphs (no modalities to
        #     weight), making it genuinely multimodal-specific.
        #   Ablation switch: graph_fusion_mode in
        #     {"dimf"(full), "gate"(no domain-inv), "add"(naive sum)}.
        # =====================================================================
        self.fusion_mode = str(getattr(config, "graph_fusion_mode", "dimf"))
        # interaction features: [text, image, text*image, |text-image|] -> gate
        self.mod_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cm_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cm_norm = nn.LayerNorm(hidden_dim)
        # domain head on the fusion gate (Innovation #3's domain-invariance).
        # predicts source domain FROM the modality-fusion signal; GRL makes the
        # fusion signal domain-agnostic.
        self.modality_grl_lambda = float(getattr(config, "graph_modality_grl_lambda", 1.0))
        self._modality_domain_logits = None
        self._modality_domain_target = None

        # token-type embedding so the LLM can tell center / neighbor / global apart
        self.semantic_type_embed = nn.Embedding(9, hidden_dim)
        nn.init.zeros_(self.semantic_type_embed.weight)

        # =====================================================================
        # Innovation #2 : PPR-reliability attention propagation + center anchor
        #   replaces the old mean-pooling "struct mix". Each layer lets the
        #   center node attend over the sub-graph with PPR as an attention
        #   prior, and writes back through a residual whose gate starts ~0 so
        #   the center's own discriminative signal is never washed out.
        # =====================================================================
        self.prop_layers = int(getattr(config, "graph_soft_token_gnn_layers", 2) or 0)
        self.prop_q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.prop_layers)])
        self.prop_k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.prop_layers)])
        self.prop_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.prop_layers)])
        self.prop_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.prop_layers)])
        # residual gate per layer, init at 0 -> identity at start of training
        self.prop_alpha = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(self.prop_layers)]
        )

        # =====================================================================
        # Innovation #1 : Domain-invariant global node (GRL + domain head)
        #   a small classifier predicts the source dataset of each *global*
        #   node; the gradient-reversal layer pushes the global representation
        #   to be domain-agnostic, killing the train-domain label-prior leak.
        # =====================================================================
        self.num_domains = int(getattr(config, "graph_num_domains", 0) or 0)
        if self.num_domains > 1:
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self.num_domains),
            )
            # domain head for the modality-fusion signal (Innovation #3 DIMF).
            # input = [g, text_summary, image_summary] compressed -> domain.
            self.modality_domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self.num_domains),
            )
        else:
            self.domain_classifier = None
            self.modality_domain_classifier = None
        self.grl_lambda = float(getattr(config, "graph_grl_lambda", 1.0))
        # last-computed domain logits/targets, picked up by the training loop
        self._domain_logits = None
        self._domain_target = None
        # Accumulated domain loss, reset by the trainer at each step and updated during each forward pass
        self._modality_domain_loss_accum = None
        self._modality_domain_loss_count = 0
        self._global_domain_loss_accum = None
        self._global_domain_loss_count = 0

        # ---- End-to-end structural prior head ----
        # Predict classes from propagated center-node features instead of an offline similarity prior.
        # Each candidate class has a learned anchor vector; its dot product with the center feature gives a logit.
        self.num_classes = int(getattr(config, "graph_num_classes", 0) or 0)
        if self.num_classes > 0:
            self.structure_prior_anchors = nn.Parameter(torch.randn(self.num_classes, hidden_dim))
            nn.init.xavier_uniform_(self.structure_prior_anchors)
            self._structure_prior_logits = None
            self._structure_prior_target = None
        else:
            self.structure_prior_anchors = None

        # ---- GCN Laplacian propagation (Enhanced) ----
        self.use_gcn = bool(getattr(config, "graph_use_gcn", True))
        self.use_gcn_1hop = bool(getattr(config, "graph_use_gcn_1hop", True))
        self.use_gcn_2hop = bool(getattr(config, "graph_use_gcn_2hop", True))
        self.use_center_protect = bool(getattr(config, "graph_use_center_protect", True))
        self.use_ppr_prop = bool(getattr(config, "graph_use_ppr_prop", True))
        if self.use_gcn:
            if self.use_gcn_1hop:
                self.gcn_1hop = GCNPropLayer(hidden_dim)
            if self.use_gcn_2hop:
                self.gcn_2hop = GCNPropLayer(hidden_dim)
            if self.use_center_protect:
                self.center_protect_gate = nn.Parameter(torch.zeros(1))

        self.output_norm = nn.LayerNorm(hidden_dim)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Resize the token-type embedding if an older checkpoint had a different
        # number of rows; everything else loads by name (unmatched keys for the
        # removed legacy modules are simply ignored).
        type_embed_key = prefix + "semantic_type_embed.weight"
        if type_embed_key in state_dict:
            loaded = state_dict[type_embed_key]
            target = self.semantic_type_embed.weight
            if loaded.shape != target.shape and loaded.ndim == 2 and loaded.shape[1] == target.shape[1]:
                resized = target.detach().clone()
                rows = min(int(loaded.shape[0]), int(target.shape[0]))
                resized[:rows] = loaded[:rows].to(dtype=resized.dtype)
                state_dict[type_embed_key] = resized
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _finite(t, clamp=8.0):
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t.clamp(min=-clamp, max=clamp) if clamp is not None else t

    def _apply_modality_masks(self, text_feat, image_feat, graph_text_mask, graph_image_mask):
        if graph_text_mask is not None:
            m = graph_text_mask.to(device=text_feat.device, dtype=torch.bool).unsqueeze(-1)
            text_feat = torch.where(
                m, self.text_mask_embed.to(text_feat.device, text_feat.dtype), text_feat
            )
        if graph_image_mask is not None:
            m = graph_image_mask.to(device=image_feat.device, dtype=torch.bool).unsqueeze(-1)
            image_feat = torch.where(
                m, self.image_mask_embed.to(image_feat.device, image_feat.dtype), image_feat
            )
        return text_feat, image_feat

    def _cross_modal_fuse(self, text_h, image_h, valid, graph_domain_id, token_type):
        """Innovation #3: Domain-Invariant Modality Fusion (DIMF).

        text_h / image_h : (B, N, H). Produces a per-token modality-fusion
        weight g in [0,1] from the cross-modal interaction features, fuses as
        g*text + (1-g)*image + interaction, and (during training) forces g to
        be domain-invariant via a gradient-reversal domain head.

        fusion_mode:
          "dimf" : full (gate + domain-invariance)            [default]
          "gate" : gate only, no domain-invariance (ablation)
          "add"  : naive sum text+image (ablation baseline)
        """
        # Match DIMF module dtype to the input during quantized or bf16 training.
        target_dtype = text_h.dtype
        if self.cm_norm.weight.dtype != target_dtype:
            self.mod_gate.to(target_dtype)
            self.cm_fuse.to(target_dtype)
            self.cm_norm.to(target_dtype)

        if self.fusion_mode == "add":
            return self.cm_norm(text_h + image_h)

        # ---- modality interaction features -> per-token fusion weight g ----
        inter = text_h * image_h
        diff = (text_h - image_h).abs()
        gate_in = torch.cat([text_h, image_h, inter, diff], dim=-1)   # (B,N,4H)
        g = torch.sigmoid(self.mod_gate(gate_in))                     # (B,N,1) in [0,1]

        # ---- fuse: weighted modalities + interaction term ----
        text_out = g * text_h
        image_out = (1.0 - g) * image_h
        fused = self.cm_fuse(torch.cat([text_out, image_out], dim=-1)) + inter
        fused = self.cm_norm(fused)

        # ---- domain-invariance on the fusion weight g (the DIMF core) ----
        if (
            self.fusion_mode == "dimf"
            and self.training
            and self.modality_domain_classifier is not None
            and graph_domain_id is not None
        ):
            if graph_domain_id.dim() == 1:
                dom = graph_domain_id.unsqueeze(1).expand_as(token_type)
            else:
                dom = graph_domain_id
            dom = dom.to(text_h.device)
            # use real (non-padded) tokens with a valid in-range domain label.
            # Domain IDs must lie in [0, num_domains) to avoid cross-entropy indexing errors.
            sel = valid & dom.ge(0) & dom.lt(self.num_domains)
            if sel.any():
                # the signal whose domain-dependence we want to remove:
                # the fusion weight g + the two modality summaries that produced it
                sig = torch.cat([g, text_h, image_h], dim=-1)[sel]    # (M, 2H+1)
                if self.modality_domain_classifier[0].weight.dtype != sig.dtype:
                    self.modality_domain_classifier.to(sig.dtype)
                rev = grad_reverse(sig, self.modality_grl_lambda)
                logits = self.modality_domain_classifier(rev)
                target = dom[sel].long()
                # Compute and accumulate the loss during each forward pass; candidate scoring invokes this function
                # multiple times, and computing it later could use a released graph.
                loss_i = F.cross_entropy(logits, target)
                if self._modality_domain_loss_accum is None:
                    self._modality_domain_loss_accum = loss_i
                    self._modality_domain_loss_count = 1
                else:
                    self._modality_domain_loss_accum = self._modality_domain_loss_accum + loss_i
                    self._modality_domain_loss_count += 1

        return fused

    def reset_domain_losses(self):
        """Reset the accumulated domain loss at the start of each training step."""
        self._modality_domain_loss_accum = None
        self._modality_domain_loss_count = 0
        self._global_domain_loss_accum = None
        self._global_domain_loss_count = 0

    def modality_domain_loss(self):
        """Mean accumulated domain-invariant loss; None if disabled or no valid tokens."""
        if getattr(self, "_modality_domain_loss_accum", None) is None:
            return None
        return self._modality_domain_loss_accum / max(1, self._modality_domain_loss_count)

    def _build_gcn_edges(self, token_type, valid, B, N):
        """Build GCN edge indices from hop structure."""
        device = valid.device
        src_1, dst_1 = [], []
        src_2, dst_2 = [], []
        for b in range(B):
            off = b * N
            hops_b = token_type[b]
            valid_b = valid[b]
            for i in range(N):
                if not valid_b[i]:
                    continue
                hi = hops_b[i].item()
                for j in range(i + 1, N):
                    if not valid_b[j]:
                        continue
                    hj = hops_b[j].item()
                    ni, nj = off + i, off + j
                    if (hi == 0 and hj == 1) or (hi == 1 and hj == 1):
                        src_1.extend([ni, nj])
                        dst_1.extend([nj, ni])
                    elif (hi == 0 and hj >= 2) or (hi == 1 and hj >= 2) or (hi >= 2 and hj >= 2):
                        src_2.extend([ni, nj])
                        dst_2.extend([nj, ni])
        e1 = torch.tensor([src_1, dst_1], dtype=torch.long, device=device) if src_1 else torch.zeros(2, 0, dtype=torch.long, device=device)
        e2 = torch.tensor([src_2, dst_2], dtype=torch.long, device=device) if src_2 else torch.zeros(2, 0, dtype=torch.long, device=device)
        return e1, e2

    def _ppr_attention_propagate(self, hidden, token_type, ppr_scores, valid):
        """Innovation #2: reliability-weighted attention propagation.

        Every node attends over all valid nodes; PPR (relevance to the center)
        is injected as an additive log-prior on the attention logits, so low-PPR
        abstract / global nodes can no longer dilute the center by brute-force
        averaging. The write-back uses a residual gate ``alpha`` initialised to
        0 (identity), letting the model *learn* how much neighbourhood to absorb.
        """
        if self.prop_layers <= 0:
            return hidden
        B, N, H = hidden.shape
        # PPR prior on the key axis: (B, 1, N)
        if ppr_scores is None:
            ppr_prior = torch.zeros(B, 1, N, device=hidden.device, dtype=hidden.dtype)
        else:
            p = ppr_scores.to(device=hidden.device, dtype=hidden.dtype).clamp(min=1e-6, max=1.0)
            ppr_prior = torch.log(p).unsqueeze(1)  # (B,1,N)
        valid_f = valid.to(hidden.dtype)
        key_mask = (~valid).unsqueeze(1)  # (B,1,N) True = ignore
        scale = 1.0 / (H ** 0.5)
        for q_lin, k_lin, v_lin, norm, alpha in zip(
            self.prop_q, self.prop_k, self.prop_v, self.prop_norm, self.prop_alpha
        ):
            q = q_lin(hidden)
            k = k_lin(hidden)
            v = v_lin(hidden)
            logits = torch.matmul(q, k.transpose(1, 2)) * scale + ppr_prior  # (B,N,N)
            logits = logits.masked_fill(key_mask, float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            agg = torch.matmul(attn, v)  # (B,N,H)
            hidden = norm(hidden + torch.tanh(alpha) * agg)
            hidden = hidden * valid_f.unsqueeze(-1)
        return hidden

    def _compute_domain_adv(self, hidden, token_type, valid, graph_domain_id):
        """Innovation #1: domain-adversarial head on the global nodes.

        Stores logits/targets on ``self`` so the training loop can add the
        cross-entropy term. Returns nothing (side-effect only). GRL makes the
        gradient flip sign, pushing global-node features to be domain-invariant.
        """
        self._domain_logits = None
        self._domain_target = None
        if self.domain_classifier is None or graph_domain_id is None:
            return
        # select global nodes (transfer / abstract) that are valid
        is_global = torch.zeros_like(token_type, dtype=torch.bool)
        for tid in self.GLOBAL_TYPE_IDS:
            is_global |= token_type.eq(tid)
        is_global &= valid
        if not is_global.any():
            return
        # graph_domain_id: (B,) or (B,N). Expand to per-node.
        if graph_domain_id.dim() == 1:
            dom = graph_domain_id.unsqueeze(1).expand_as(token_type)
        else:
            dom = graph_domain_id
        dom = dom.to(hidden.device)
        # Only nodes that carry a *valid* source-domain label (>=0) are
        # adversarial targets. The global prototype itself is tagged -1 (no
        # single domain) and is correctly excluded here.
        # Exclude domain IDs outside [0, num_domains) to prevent cross-entropy indexing errors.
        is_global &= dom.ge(0) & dom.lt(self.num_domains)
        if not is_global.any():
            return
        feats = hidden[is_global]                      # (M, H)
        if self.domain_classifier[0].weight.dtype != feats.dtype:
            self.domain_classifier.to(feats.dtype)
        rev = grad_reverse(feats, self.grl_lambda)     # gradient reversal
        logits = self.domain_classifier(rev)           # (M, num_domains)
        target = dom[is_global].long()
        # Compute and accumulate the loss immediately to avoid using a released graph.
        loss_i = F.cross_entropy(logits, target)
        if getattr(self, "_global_domain_loss_accum", None) is None:
            self._global_domain_loss_accum = loss_i
            self._global_domain_loss_count = 1
        else:
            self._global_domain_loss_accum = self._global_domain_loss_accum + loss_i
            self._global_domain_loss_count += 1

    def domain_adversarial_loss(self):
        """Mean accumulated global-node domain adversarial loss; None if disabled."""
        if getattr(self, "_global_domain_loss_accum", None) is None:
            return None
        return self._global_domain_loss_accum / max(1, self._global_domain_loss_count)

    def structure_prior_loss(self, graph_label_id):
        """End-to-end structural prior loss from center-node features to class logits.
        graph_label_id: (B,) true class ID for each sample; -1 means ignore.
        """
        if self._structure_prior_logits is None or graph_label_id is None:
            return None
        target = graph_label_id.to(self._structure_prior_logits.device)
        sel = target.ge(0)
        if not sel.any():
            return None
        return F.cross_entropy(self._structure_prior_logits[sel], target[sel])

    def forward(
        self,
        graph_emb,
        embed_tokens,
        graph_node_mask=None,
        graph_hops=None,
        graph_degrees=None,
        graph_ppr_scores=None,
        graph_role_ids=None,
        graph_role_mask=None,
        graph_text_mask=None,
        graph_image_mask=None,
        graph_domain_id=None,
        graph_label_id=None,
    ):
        if isinstance(self.text_projector, nn.Sequential):
            module_dtype = self.text_projector[0].weight.dtype
        else:
            module_dtype = self.text_projector.weight.dtype
        # Match all encoder submodules to the main dtype; modules added during 4-bit/bf16 training
        # such as DIMF, propagation, and output normalization may otherwise remain in fp32.
        if self.output_norm.weight.dtype != module_dtype:
            self.to(module_dtype)
        graph_emb = graph_emb.to(module_dtype)
        B, N = graph_emb.shape[:2]

        # split CLIP embedding into text / image halves
        text_feat, image_feat = graph_emb.split(self.modal_dim, dim=-1)
        text_feat = self._finite(text_feat)
        image_feat = self._finite(image_feat)
        text_feat, image_feat = self._apply_modality_masks(
            text_feat, image_feat, graph_text_mask, graph_image_mask
        )

        # token type & validity
        if graph_hops is not None:
            token_type = graph_hops.to(device=graph_emb.device, dtype=torch.long).clamp(0, 8)
        else:
            token_type = torch.zeros(B, N, dtype=torch.long, device=graph_emb.device)
        if graph_node_mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=graph_emb.device)
        else:
            valid = graph_node_mask.to(device=graph_emb.device, dtype=torch.bool)

        # project each modality into LLM hidden space
        text_h = self.text_projector(text_feat)
        image_h = self.image_projector(image_feat)

        # ---- Innovation #3: cross-modal complementary fusion ----
        fused = self._cross_modal_fuse(text_h, image_h, valid, graph_domain_id, token_type)
        fused = fused + self.semantic_type_embed(token_type)
        fused = self._finite(fused)
        fused = fused * valid.unsqueeze(-1).to(fused.dtype)

        # ---- Innovation #2: PPR-reliability attention propagation ----
        if getattr(self, 'use_ppr_prop', True):
            hidden = self._ppr_attention_propagate(fused, token_type, graph_ppr_scores, valid)
        else:
            hidden = fused

        # ---- Enhanced: GCN Laplacian propagation ----
        if getattr(self, 'use_gcn', True):
            center_feat_pre = hidden[:, 0, :].clone()

            edge_1hop, edge_2hop = self._build_gcn_edges(token_type, valid, B, N)
            H_flat = hidden.reshape(B * N, -1)

            if getattr(self, 'use_gcn_1hop', True) and hasattr(self, 'gcn_1hop') and edge_1hop.shape[1] > 0:
                H_flat = self.gcn_1hop(H_flat, edge_1hop, B * N)
            if getattr(self, 'use_gcn_2hop', True) and hasattr(self, 'gcn_2hop') and edge_2hop.shape[1] > 0:
                H_flat = self.gcn_2hop(H_flat, edge_2hop, B * N)

            hidden_gcn = H_flat.reshape(B, N, -1)
            hidden_gcn = hidden_gcn * valid.unsqueeze(-1).to(hidden_gcn.dtype)

            # ---- Center-protection gating ----
            if getattr(self, 'use_center_protect', True) and hasattr(self, 'center_protect_gate'):
                gate = torch.sigmoid(self.center_protect_gate)
                center_feat_post = hidden_gcn[:, 0, :]
                protected_center = (1.0 - gate) * center_feat_pre + gate * center_feat_post
                hidden_gcn = torch.cat([protected_center.unsqueeze(1), hidden_gcn[:, 1:, :]], dim=1)

            hidden = hidden_gcn

        # ---- Innovation #1: domain-adversarial signal on global nodes ----
        self._compute_domain_adv(hidden, token_type, valid, graph_domain_id)

        # ---- End-to-end structural prior: center-node features to class logits ----
        self._structure_prior_logits = None
        if self.structure_prior_anchors is not None and self.training:
            center_feat = hidden[:, 0, :]                      # (B, H)
            self._structure_prior_logits = center_feat @ self.structure_prior_anchors.t()  # (B, num_classes)

        graph_hidden = self.output_norm(hidden)
        graph_hidden = self._finite(graph_hidden)
        graph_hidden = graph_hidden.masked_fill(~valid.unsqueeze(-1), 0)
        return graph_hidden


class PlainGCNTokenEncoder(nn.Module):
    """Plain baseline: ordinary GCN output mapped to the LLM graph tokens.

    Identical interface/inputs/outputs to ``GraphSoftTokenEncoder`` so the whole
    downstream pipeline (frozen Vicuna, same textual evidence and candidate
    labels) is unchanged; only the token construction is a plain GCN.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        clip_dim = int(getattr(config, "graph_hidden_size"))
        self.modal_dim = clip_dim // 2
        hidden_dim = int(getattr(config, "hidden_size"))
        self.hidden_dim = hidden_dim
        # learnable missing-modality placeholders (same role as CHARM's)
        self.text_mask_embed = nn.Parameter(torch.zeros(self.modal_dim))
        self.image_mask_embed = nn.Parameter(torch.zeros(self.modal_dim))
        # plain fusion: concatenated text|image -> LLM hidden space
        self.in_proj = nn.Linear(clip_dim, hidden_dim)
        # plain GCN layers over the retrieved subgraph (no attention, no gates)
        self.gcn_1hop = GCNPropLayer(hidden_dim)
        self.gcn_2hop = GCNPropLayer(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _finite(t, clamp=8.0):
        return torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)

    def _apply_modality_masks(self, text_feat, image_feat, text_mask, image_mask):
        if text_mask is not None:
            m = text_mask.to(device=text_feat.device, dtype=torch.bool).unsqueeze(-1)
            text_feat = torch.where(m, self.text_mask_embed.to(text_feat.device, text_feat.dtype), text_feat)
        if image_mask is not None:
            m = image_mask.to(device=image_feat.device, dtype=torch.bool).unsqueeze(-1)
            image_feat = torch.where(m, self.image_mask_embed.to(image_feat.device, image_feat.dtype), image_feat)
        return text_feat, image_feat

    @staticmethod
    def _build_edges(token_type, valid, B, N, device):
        src_1, dst_1, src_2, dst_2 = [], [], [], []
        for b in range(B):
            off = b * N
            hb = token_type[b]
            vb = valid[b]
            for i in range(N):
                if not bool(vb[i]):
                    continue
                hi = int(hb[i].item())
                for j in range(i + 1, N):
                    if not bool(vb[j]):
                        continue
                    hj = int(hb[j].item())
                    ni, nj = off + i, off + j
                    if (hi == 0 and hj == 1) or (hi == 1 and hj == 1):
                        src_1.extend([ni, nj]); dst_1.extend([nj, ni])
                    else:
                        src_2.extend([ni, nj]); dst_2.extend([nj, ni])
        e1 = torch.tensor([src_1, dst_1], dtype=torch.long, device=device) if src_1             else torch.zeros(2, 0, dtype=torch.long, device=device)
        e2 = torch.tensor([src_2, dst_2], dtype=torch.long, device=device) if src_2             else torch.zeros(2, 0, dtype=torch.long, device=device)
        return e1, e2

    def forward(self, graph_emb, embed_tokens, graph_node_mask=None, graph_hops=None,
                graph_degrees=None, graph_ppr_scores=None, graph_role_ids=None,
                graph_role_mask=None, graph_text_mask=None, graph_image_mask=None,
                graph_domain_id=None, graph_label_id=None):
        module_dtype = self.in_proj.weight.dtype
        if self.output_norm.weight.dtype != module_dtype:
            self.to(module_dtype)
        graph_emb = graph_emb.to(module_dtype)
        B, N = graph_emb.shape[:2]
        if graph_node_mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=graph_emb.device)
        else:
            valid = graph_node_mask.to(device=graph_emb.device, dtype=torch.bool)

        text_feat, image_feat = graph_emb.split(self.modal_dim, dim=-1)
        text_feat = self._finite(text_feat)
        image_feat = self._finite(image_feat)
        text_feat, image_feat = self._apply_modality_masks(text_feat, image_feat, graph_text_mask, graph_image_mask)

        hidden = self.in_proj(torch.cat([text_feat, image_feat], dim=-1))
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)

        if graph_hops is not None:
            token_type = graph_hops.to(device=hidden.device, dtype=torch.long).clamp(0, 8)
        else:
            token_type = torch.zeros(B, N, dtype=torch.long, device=hidden.device)

        e1, e2 = self._build_edges(token_type, valid, B, N, hidden.device)
        H_flat = hidden.reshape(B * N, -1)
        if e1.shape[1] > 0:
            H_flat = self.gcn_1hop(H_flat, e1, B * N)
        if e2.shape[1] > 0:
            H_flat = self.gcn_2hop(H_flat, e2, B * N)
        hidden = H_flat.reshape(B, N, -1)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)

        out = self._finite(self.output_norm(hidden))
        return out.masked_fill(~valid.unsqueeze(-1), 0)


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

        if hasattr(config, "graph_hidden_size"):
            if getattr(config, "graph_use_soft_tokens", False):
                self.graph_soft_token_encoder = (PlainGCNTokenEncoder(config) if getattr(config, 'graph_soft_token_encoder_type', 'charm') == 'plain_gcn' else GraphSoftTokenEncoder(config))
            else:
                self.graph_projector = build_graph_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            # Also load graph_soft_token_encoder weights if present
            graph_encoder_weights = get_w(mm_projector_weights, 'graph_soft_token_encoder')
            if graph_encoder_weights and hasattr(self, 'graph_soft_token_encoder'):
                self.graph_soft_token_encoder.load_state_dict(graph_encoder_weights, strict=False)
                print(f"Loaded {len(graph_encoder_weights)} graph_soft_token_encoder params from {pretrain_mm_mlp_adapter}")

    def initialize_graph_modules(self, model_args):
        graph_hidden_size = getattr(model_args, "graph_hidden_size", None)
        if graph_hidden_size is None:
            return

        self.config.use_graph_proj = True
        self.config.graph_hidden_size = graph_hidden_size
        self.config.graph_use_soft_tokens = getattr(model_args, "graph_use_soft_tokens", False)
        if self.config.graph_use_soft_tokens:
            self.config.graph_projector_input_size = graph_hidden_size // 2
        self.config.graph_projector_type = getattr(model_args, "graph_projector_type", "linear")
        self.config.graph_modality_mask_prob = getattr(model_args, "graph_modality_mask_prob", 0.15)
        self.config.graph_disable_role_tokens = getattr(model_args, "graph_disable_role_tokens", False)
        # number of PPR-reliability attention-propagation layers (Innovation #2)
        self.config.graph_soft_token_gnn_layers = getattr(model_args, "graph_soft_token_gnn_layers", 2)
        # cross-modal fusion / propagation attention heads (Innovation #3)
        self.config.graph_attn_heads = getattr(model_args, "graph_attn_heads", 8)
        # fusion mode (Innovation #3 ablation): "cross_attn"(full) | "add"(baseline)
        self.config.graph_fusion_mode = getattr(model_args, "graph_fusion_mode", "dimf")
        self.config.graph_modality_grl_lambda = getattr(model_args, "graph_modality_grl_lambda", 1.0)
        # domain-adversarial global node (Innovation #1):
        #   set graph_num_domains = number of training datasets to enable.
        self.config.graph_num_domains = getattr(model_args, "graph_num_domains", 0)
        self.config.graph_grl_lambda = getattr(model_args, "graph_grl_lambda", 1.0)
        self.config.graph_num_classes = getattr(model_args, "graph_num_classes", 0)
        self.config.graph_structure_prior_loss_weight = getattr(model_args, "graph_structure_prior_loss_weight", 0.0)
        self.config.graph_soft_token_encoder_type = getattr(model_args, "graph_soft_token_encoder_type", "charm")

        if self.config.graph_use_soft_tokens:
            if getattr(self, "graph_soft_token_encoder", None) is None:
                self.graph_soft_token_encoder = (PlainGCNTokenEncoder(self.config) if getattr(self.config, 'graph_soft_token_encoder_type', 'charm') == 'plain_gcn' else GraphSoftTokenEncoder(self.config))
            else:
                for p in self.graph_soft_token_encoder.parameters():
                    p.requires_grad = True
        else:
            if getattr(self, "graph_projector", None) is None:
                self.graph_projector = build_graph_projector(self.config)
            else:
                for p in self.graph_projector.parameters():
                    p.requires_grad = True


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        projector_dtype = next(self.get_model().mm_projector.parameters()).dtype
        image_features = image_features.to(projector_dtype)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_graphs(
        self,
        graph_emb,
        graph_node_mask=None,
        graph_hops=None,
        graph_degrees=None,
        graph_ppr_scores=None,
        graph_role_ids=None,
        graph_role_mask=None,
        graph_text_mask=None,
        graph_image_mask=None,
        graph_domain_id=None,
    ):
        # ---- Token-type ablation: mask out specific hop types before encoding ----
        # graph_ablate_hop: legacy hop-level mask
        ablate_hop = int(getattr(self.config, "graph_ablate_hop", -1))
        if ablate_hop >= 0 and graph_hops is not None and graph_node_mask is not None:
            ablate_mask = graph_hops.eq(ablate_hop).to(graph_node_mask.device)
            graph_node_mask = graph_node_mask.clone()
            graph_node_mask[ablate_mask] = False
        # graph_ablate_layer: training-time layer ablation (applied at data level, but also mask here for safety)
        # hop mapping: layer1=2, layer2=5, transfer=4
        ablate_layer = int(getattr(self.config, "graph_ablate_layer", -1))
        if ablate_layer >= 0 and graph_hops is not None and graph_node_mask is not None:
            hop_map = {1: 2, 2: 5, 4: 4}
            ablate_hop2 = hop_map.get(ablate_layer, -1)
            if ablate_hop2 >= 0:
                ablate_mask2 = graph_hops.eq(ablate_hop2).to(graph_node_mask.device)
                graph_node_mask = graph_node_mask.clone()
                graph_node_mask[ablate_mask2] = False

        if getattr(self.config, "graph_use_soft_tokens", False) and hasattr(self.get_model(), "graph_soft_token_encoder"):
            return self.get_model().graph_soft_token_encoder(
                graph_emb=graph_emb,
                embed_tokens=self.get_model().embed_tokens,
                graph_node_mask=graph_node_mask,
                graph_hops=graph_hops,
                graph_degrees=graph_degrees,
                graph_ppr_scores=graph_ppr_scores,
                graph_role_ids=graph_role_ids,
                graph_role_mask=graph_role_mask,
                graph_text_mask=graph_text_mask,
                graph_image_mask=graph_image_mask,
                graph_domain_id=graph_domain_id,
            )

        projector_dtype = next(self.get_model().graph_projector.parameters()).dtype
        graph_features = self.get_model().graph_projector(graph_emb.to(projector_dtype))
        if graph_node_mask is not None:
            graph_features = graph_features.clone()
            graph_features[~graph_node_mask] = 0
        return graph_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, graphs=None, graph_emb=None, graph_node_mask=None,
        graph_hops=None, graph_degrees=None, graph_ppr_scores=None,
        graph_role_ids=None, graph_role_mask=None,
        graph_text_mask=None, graph_image_mask=None,
        graph_domain_id=None,
        return_graph_meta=False
    ):
        vision_tower = self.get_vision_tower()
        has_images = vision_tower is not None and images is not None and input_ids.shape[1] != 1
        has_graphs = graph_emb is not None and input_ids.shape[1] != 1
        if not has_images and not has_graphs:
            if return_graph_meta:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if has_images and (type(images) is list or images.ndim == 5):
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        elif has_images:
            image_features = self.encode_images(images)
        else:
            image_features = None

        if has_graphs:
            if graph_node_mask is None:
                if graphs is not None:
                    graph_node_mask = graphs.ne(DEFAULT_GRAPH_PAD_ID)
                else:
                    graph_node_mask = torch.ones(graph_emb.shape[:2], dtype=torch.bool, device=graph_emb.device)
            graph_features = self.encode_graphs(
                graph_emb,
                graph_node_mask=graph_node_mask,
                graph_hops=graph_hops,
                graph_degrees=graph_degrees,
                graph_ppr_scores=graph_ppr_scores,
                graph_role_ids=graph_role_ids,
                graph_role_mask=graph_role_mask,
                graph_text_mask=graph_text_mask,
                graph_image_mask=graph_image_mask,
                graph_domain_id=graph_domain_id,
            )
            return self._prepare_inputs_with_graph(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                image_features=image_features,
                graph_features=graph_features,
                graph_node_mask=graph_node_mask,
                return_graph_meta=return_graph_meta,
            )

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if return_graph_meta:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def _prepare_inputs_with_graph(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        image_features,
        graph_features,
        graph_node_mask,
        return_graph_meta=False,
    ):
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        if image_features is not None:
            if isinstance(image_features, torch.Tensor):
                image_feature_list = [image_features[i] for i in range(image_features.shape[0])]
            else:
                image_feature_list = list(image_features)
        else:
            image_feature_list = [None] * len(input_ids)

        graph_feature_list = []
        for feats, mask in zip(graph_features, graph_node_mask):
            if mask.dim() == 1:
                valid_mask = mask.to(feats.device)
                if valid_mask.any():
                    graph_feature_list.append([feats[valid_mask]])
                else:
                    graph_feature_list.append([])
            else:
                sample_graphs = []
                for cur_feats, cur_mask in zip(feats, mask):
                    valid_mask = cur_mask.to(cur_feats.device)
                    if valid_mask.any():
                        sample_graphs.append(cur_feats[valid_mask])
                graph_feature_list.append(sample_graphs)

        new_input_embeds = []
        new_labels = []
        graph_meta = []

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_labels = labels[batch_idx]
            cur_image_features = image_feature_list[batch_idx]
            if not isinstance(cur_image_features, (list, tuple)):
                cur_image_features = [cur_image_features] if cur_image_features is not None else None
            cur_graph_features_list = graph_feature_list[batch_idx]

            image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            graph_positions = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0].tolist()
            # Rich prompt style: allow multiple <image> tokens (center + PPR neighbors)
            if len(graph_positions) != len(cur_graph_features_list):
                raise ValueError(
                    f"Number of <graph> placeholders ({len(graph_positions)}) does not match provided graph spans ({len(cur_graph_features_list)})."
                )

            special_positions = [(pos, "image") for pos in image_positions] + [(pos, "graph") for pos in graph_positions]
            special_positions.sort(key=lambda x: x[0])

            cur_new_input_embeds = []
            cur_new_labels = []
            cursor = 0
            current_len = 0
            current_graph_meta = []
            graph_idx = 0

            for pos, kind in special_positions:
                if pos > cursor:
                    text_ids = cur_input_ids[cursor:pos]
                    text_labels = cur_labels[cursor:pos]
                    cur_new_input_embeds.append(self.get_model().embed_tokens(text_ids))
                    cur_new_labels.append(text_labels)
                    current_len += text_ids.shape[0]

                if kind == "image":
                    if cur_image_features is None or len(cur_image_features) == 0:
                        raise ValueError("Encountered <image> token but image features are missing.")
                    img_idx = image_positions.index(pos) if hasattr(image_positions, 'index') else 0
                    cur_img = cur_image_features[img_idx] if isinstance(cur_image_features, (list, tuple)) and img_idx < len(cur_image_features) else cur_image_features[0]
                    cur_new_input_embeds.append(cur_img)
                    cur_new_labels.append(
                        torch.full(
                            (cur_img.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    current_len += cur_img.shape[0]
                else:
                    cur_graph_features = cur_graph_features_list[graph_idx]
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_graph_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    current_graph_meta.append({
                        "start": current_len,
                        "length": cur_graph_features.shape[0],
                        "graph_index": graph_idx,
                    })
                    current_len += cur_graph_features.shape[0]
                    graph_idx += 1

                cursor = pos + 1

            if cursor < cur_input_ids.shape[0]:
                text_ids = cur_input_ids[cursor:]
                text_labels = cur_labels[cursor:]
                cur_new_input_embeds.append(self.get_model().embed_tokens(text_ids))
                cur_new_labels.append(text_labels)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            graph_meta.append(current_graph_meta)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            truncated_input_embeds = []
            truncated_labels = []
            truncated_graph_meta = []
            for cur_embed, cur_label, cur_graph_meta in zip(new_input_embeds, new_labels, graph_meta):
                cur_embed = cur_embed[:tokenizer_model_max_length]
                cur_label = cur_label[:tokenizer_model_max_length]
                truncated_meta = []
                for meta in cur_graph_meta:
                    if meta["start"] >= tokenizer_model_max_length:
                        continue
                    truncated_meta.append(
                        {
                            "start": meta["start"],
                            "length": min(meta["length"], tokenizer_model_max_length - meta["start"]),
                            "graph_index": meta.get("graph_index", 0),
                        }
                    )
                truncated_input_embeds.append(cur_embed)
                truncated_labels.append(cur_label)
                truncated_graph_meta.append(truncated_meta)
            new_input_embeds = truncated_input_embeds
            new_labels = truncated_labels
            graph_meta = truncated_graph_meta

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                pad_len = max_len - cur_len
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros((pad_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                for meta in graph_meta[i]:
                    meta["start"] += pad_len
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if return_graph_meta:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, graph_meta
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_graph_tokens = tokenizer.add_tokens([DEFAULT_GRAPH_TOKEN], special_tokens=True)
        if num_new_graph_tokens > 0:
            self.resize_token_embeddings(len(tokenizer))
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            input_embeddings[-num_new_graph_tokens:] = input_embeddings[:-num_new_graph_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_graph_tokens:] = output_embeddings[:-num_new_graph_tokens].mean(dim=0, keepdim=True)

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
