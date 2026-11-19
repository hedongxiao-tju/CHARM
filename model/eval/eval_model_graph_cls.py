import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN
from model.model.builder import load_pretrained_model
from model.mm_utils import get_model_name_from_path, tokenizer_graph_image_token
from model.utils import disable_torch_init


DEFAULT_MODEL_BASE = "models/vicuna-7b-v1.5"
DEFAULT_DATA_ROOT = "data/ele-fashion"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--classifier-epochs", type=int, default=300)
    parser.add_argument("--classifier-lr", type=float, default=1e-2)
    parser.add_argument("--classifier-weight-decay", type=float, default=5e-4)
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-sample-size", type=int, default=10)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def load_graph_caches(data_root: str, graph_hops: int, graph_sample_size: int) -> Tuple[torch.Tensor, np.memmap]:
    graph_ids_path = os.path.join(data_root, f"graph_ids_h{graph_hops}_s{graph_sample_size}.pt")
    graph_feat_path = os.path.join(data_root, f"graph_feat_h{graph_hops}_s{graph_sample_size}_fp16.npy")
    graph_ids = torch.load(graph_ids_path, map_location="cpu", weights_only=False).to(torch.long)
    graph_feats = np.load(graph_feat_path, mmap_mode="r")
    return graph_ids, graph_feats


def load_labels_and_masks(data_root: str):
    labels = torch.load(os.path.join(data_root, "labels.pt"), map_location="cpu", weights_only=False)
    train_mask = torch.load(os.path.join(data_root, "train_mask.pt"), map_location="cpu", weights_only=False).bool()
    val_mask = torch.load(os.path.join(data_root, "val_mask.pt"), map_location="cpu", weights_only=False).bool()
    test_mask = torch.load(os.path.join(data_root, "test_mask.pt"), map_location="cpu", weights_only=False).bool()
    return labels, train_mask, val_mask, test_mask


def slice_dataset(labels, train_mask, val_mask, test_mask, graph_ids, graph_feats, max_samples: int):
    if max_samples <= 0:
        return labels, train_mask, val_mask, test_mask, graph_ids, graph_feats
    labels = labels[:max_samples]
    train_mask = train_mask[:max_samples]
    val_mask = val_mask[:max_samples]
    test_mask = test_mask[:max_samples]
    graph_ids = graph_ids[:max_samples]
    graph_feats = graph_feats[:max_samples]
    return labels, train_mask, val_mask, test_mask, graph_ids, graph_feats


def build_prompt_input_ids(tokenizer, device):
    prompt = (
        f"{DEFAULT_GRAPH_TOKEN}\n"
        "Please briefly describe the center fashion product by using the BFS-serialized neighborhood graph."
    )
    input_ids = tokenizer_graph_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    return input_ids


@torch.no_grad()
def extract_graph_embeddings(model, tokenizer, graph_ids, graph_feats, batch_size: int, device: torch.device):
    model.eval()
    prompt_input_ids = build_prompt_input_ids(tokenizer, device)
    all_embeds = []

    for start in range(0, graph_ids.shape[0], batch_size):
        end = min(start + batch_size, graph_ids.shape[0])
        batch_graph_ids = graph_ids[start:end].to(device)
        batch_graph_emb = torch.from_numpy(np.array(graph_feats[start:end], copy=True)).to(device=device, dtype=model.dtype)
        batch_graph_mask = batch_graph_ids.ne(DEFAULT_GRAPH_PAD_ID)
        batch_input_ids = prompt_input_ids.expand(end - start, -1)

        (
            _,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            graph_meta,
        ) = model.prepare_inputs_labels_for_multimodal(
            batch_input_ids,
            None,
            None,
            None,
            None,
            None,
            image_sizes=None,
            graphs=batch_graph_ids,
            graph_emb=batch_graph_emb,
            graph_node_mask=batch_graph_mask,
            return_graph_meta=True,
        )

        outputs = model.get_model()(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        batch_embeds = []
        for sample_hidden, sample_meta in zip(hidden_states, graph_meta):
            if sample_meta is None or sample_meta["length"] <= 0:
                batch_embeds.append(torch.zeros(hidden_states.shape[-1], device=hidden_states.device, dtype=torch.float32))
                continue
            start_idx = sample_meta["start"]
            length = sample_meta["length"]
            graph_hidden = sample_hidden[start_idx:start_idx + length]
            batch_embeds.append(graph_hidden.float().mean(dim=0))
        all_embeds.append(torch.stack(batch_embeds, dim=0).cpu())

    return torch.cat(all_embeds, dim=0)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_linear_probe(embeddings, labels, train_mask, val_mask, test_mask, epochs, lr, weight_decay):
    device = embeddings.device
    num_classes = int(labels.max().item() + 1)
    clf = LinearClassifier(embeddings.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_test = -1.0
    best_state = None

    for _ in range(epochs):
        clf.train()
        logits = clf(embeddings)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            logits = clf(embeddings)
            val_acc = (logits[val_mask].argmax(dim=1) == labels[val_mask]).float().mean().item()
            test_acc = (logits[test_mask].argmax(dim=1) == labels[test_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)

    return best_val, best_test


def main():
    args = parse_args()
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model_device = next(model.parameters()).device

    labels, train_mask, val_mask, test_mask = load_labels_and_masks(args.data_root)
    graph_ids, graph_feats = load_graph_caches(args.data_root, args.graph_hops, args.graph_sample_size)
    labels, train_mask, val_mask, test_mask, graph_ids, graph_feats = slice_dataset(
        labels, train_mask, val_mask, test_mask, graph_ids, graph_feats, args.max_samples
    )
    embeddings = extract_graph_embeddings(
        model=model,
        tokenizer=tokenizer,
        graph_ids=graph_ids,
        graph_feats=graph_feats,
        batch_size=args.batch_size,
        device=model_device,
    )

    embeddings = embeddings.to(model_device)
    labels = labels.to(model_device)
    train_mask = train_mask.to(model_device)
    val_mask = val_mask.to(model_device)
    test_mask = test_mask.to(model_device)

    best_val, best_test = train_linear_probe(
        embeddings=embeddings,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs=args.classifier_epochs,
        lr=args.classifier_lr,
        weight_decay=args.classifier_weight_decay,
    )

    print("=" * 60)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Test accuracy: {best_test:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
