"""CHARM joint node-classification/link-prediction training entrypoint.

This wrapper keeps the existing LLaVA trainer unchanged and injects a mixed
dataset at runtime.  Set CHARM_TASKS to a comma-separated task list aligned
with --graph_data_roots, for example: nc,nc,lp,lp.
"""
import os
from dataclasses import replace
from typing import Dict, List

import torch

import model.train.train as train_mod
from model.constants import DEFAULT_GRAPH_TOKEN, IGNORE_INDEX
from model.train.train import (
    DataCollatorForSupervisedDataset,
    LazyGraphSupervisedDataset,
    build_multi_graph_data_args,
    preprocess,
)


def _task_list(data_args):
    roots = str(data_args.graph_data_roots).split(",")
    tasks = [x.strip().lower() for x in os.environ.get("CHARM_TASKS", "").split(",") if x.strip()]
    if not tasks:
        tasks = ["nc"] * len(roots)
    if len(tasks) != len(roots):
        raise ValueError(f"CHARM_TASKS has {len(tasks)} entries, expected {len(roots)}")
    if any(task not in {"nc", "lp"} for task in tasks):
        raise ValueError(f"Unsupported CHARM task list: {tasks}")
    return tasks


def _encode_prompt(tokenizer, prompt, answer):
    sources = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]]
    return preprocess(sources, tokenizer, has_image=False, has_graph=True)


class CharmNCDataset(LazyGraphSupervisedDataset):
    """Existing graph samples with a compact, continuous CHARM prompt."""

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        sample_idx = self.sample_indices[i]
        record = self.records[sample_idx]
        label_id = int(self.labels[sample_idx].item())
        label = self.graph_label_spec["valid_id2label"][label_id]
        labels = ", ".join(self.graph_label_spec["candidate_labels"])
        text = self._get_record_text(record)
        prompt = (
            f"{DEFAULT_GRAPH_TOKEN}\n"
            "Graph tokens encode the center product and its context. "
            "Token order: center-text, center-image, layer-1 abstract, layer-2 abstract, "
            "calibration, transfer, then retrieved structural neighbors. "
            "Task: classify the center product. "
            f"Center: {text} "
            f"Categories: {labels}. "
            "Answer with one category name."
        )
        encoded = _encode_prompt(self.tokenizer, prompt, label)
        sample["input_ids"] = encoded["input_ids"][0]
        sample["labels"] = encoded["labels"][0]
        return sample


class CharmLPDataset(torch.utils.data.Dataset):
    """Pair samples built from the same node graph encoder used by CHARM NC."""

    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.nodes = CharmNCDataset(tokenizer=tokenizer, data_args=replace(data_args, graph_task="nc"))
        root = data_args.graph_data_root
        split_path = os.path.join(root, "lp_zeroshot_split.pt")
        if not os.path.exists(split_path):
            split_path = os.path.join(root, "lp_edge_split.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"No LP split found under {root}")
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        self.samples = self._read_split(split)
        limit = int(os.environ.get("CHARM_MAX_LP_SAMPLES", "5000"))
        if limit > 0:
            self.samples = self.samples[:limit]

    @staticmethod
    def _read_split(split):
        if "train" not in split:
            raise ValueError("LP split has no train section")
        train = split["train"]
        if "pos_edges" in train:
            out = [(int(x[0]), int(x[1]), 1) for x in train["pos_edges"]]
            out += [(int(x[0]), int(x[1]), 0) for x in train["neg_edges"]]
            return out
        if isinstance(train, list):
            return [(int(x["u"]), int(x["v"]), int(x["label"])) for x in train]
        raise ValueError(f"Unsupported LP train split keys: {list(train)}")

    def __len__(self):
        return len(self.samples)

    def _node(self, node_id):
        self.nodes.sample_indices = [node_id]
        return self.nodes[0]

    def __getitem__(self, i):
        u, v, label = self.samples[i]
        left = self._node(u)
        right = self._node(v)
        graph = torch.cat([left["graph"], right["graph"]], dim=0)
        graph_emb = torch.cat([left["graph_emb"], right["graph_emb"]], dim=0)
        graph_mask = torch.cat([left["graph_node_mask"], right["graph_node_mask"]], dim=0)
        graph_hops = torch.cat([left["graph_hops"], right["graph_hops"]], dim=0)
        role_ids = left["graph_role_ids"] + right["graph_role_ids"]
        text_u = self.nodes._get_record_text(self.nodes.records[u])
        text_v = self.nodes._get_record_text(self.nodes.records[v])
        prompt = (
            f"{DEFAULT_GRAPH_TOKEN}\n"
            "The graph contains two consecutive node contexts. In each context "
            "the tokens are center-text, center-image, layer-1 abstract, layer-2 abstract, "
            "calibration, and transfer, followed by structural neighbors. "
            "Task: predict whether the two center products are linked. "
            f"Center A: {text_u} "
            f"Center B: {text_v} "
            "Answer yes or no."
        )
        answer = "yes" if label else "no"
        encoded = _encode_prompt(self.tokenizer, prompt, answer)
        return {
            "input_ids": encoded["input_ids"][0],
            "labels": encoded["labels"][0],
            "graph": graph,
            "graph_emb": graph_emb,
            "graph_targets": graph_emb.clone(),
            "graph_node_mask": graph_mask,
            "graph_hops": graph_hops,
            "graph_role_ids": role_ids,
        }


def charm_data_module(tokenizer, data_args) -> Dict:
    tasks = _task_list(data_args)
    graph_args = build_multi_graph_data_args(data_args)
    datasets = []
    for task, args in zip(tasks, graph_args):
        datasets.append(
            CharmNCDataset(tokenizer, args) if task == "nc" else CharmLPDataset(tokenizer, args)
        )
    dataset = train_mod.LengthAwareConcatDataset(datasets)
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        graph_modality_mask_prob=0.0,
        graph_modality_mask_strategy="none",
    )
    print(f"[CHARM] joint datasets={len(datasets)}, samples={len(dataset)}, tasks={tasks}")
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": collator}


def main():
    train_mod.make_supervised_data_module = charm_data_module
    train_mod.train()


if __name__ == "__main__":
    main()
