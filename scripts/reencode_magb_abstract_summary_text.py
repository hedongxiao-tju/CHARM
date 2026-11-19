#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


NODE_FILES = [
    "layer1_abstract_nodes.pt",
    "layer2_abstract_nodes.pt",
    "global_semantic_prototype_nodes.pt",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Encode rewritten MAGB abstract summaries with the full CLIP tokenizer/model.")
    parser.add_argument("--input-root", default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic")
    parser.add_argument("--output-root", default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic_cliptext")
    parser.add_argument("--model-path", default="models/clip-vit-large-patch14-336-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-limit-gb", type=float, default=20.0)
    parser.add_argument("--estimate-only", action="store_true")
    return parser.parse_args()


def estimate_peak_gb(input_root: Path, model_path: Path):
    pt_bytes = sum((input_root / name).stat().st_size for name in NODE_FILES if (input_root / name).exists())
    model_bytes = sum(path.stat().st_size for path in model_path.glob("*.safetensors"))
    tokenizer_bytes = sum(path.stat().st_size for path in model_path.glob("*.json")) + sum(path.stat().st_size for path in model_path.glob("*.txt"))
    return (pt_bytes * 2.0 + model_bytes * 2.5 + tokenizer_bytes * 4.0 + 1024**3) / (1024**3)


def copy_tree(input_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    for path in input_root.iterdir():
        if path.is_file():
            shutil.copy2(path, output_root / path.name)


def validate_text_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("CLIPProcessor did not load a tokenizer.")
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    if vocab_size < 1000:
        raise RuntimeError(
            "Invalid CLIP tokenizer: vocab is too small. "
            f"vocab_size={vocab_size}"
        )
    probe = tokenizer(
        ["fresh organic fruit snacks", "science fiction movie collections"],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )["input_ids"]
    if torch.equal(probe[0], probe[1]):
        raise RuntimeError("Invalid CLIP tokenizer: distinct probe texts produced identical token ids.")


def encode_texts(model, processor, texts, batch_size, device):
    out = []
    for start in range(0, len(texts), batch_size):
        batch = [text if text else "abstract semantic node" for text in texts[start : start + batch_size]]
        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            text_outputs = model.text_model(**inputs)
            feats = model.text_projection(text_outputs.pooler_output)
            feats = F.normalize(feats.float(), dim=-1)
        out.append(feats.cpu())
        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(out, dim=0)


def replace_embeddings(nodes, embeddings, offset=0):
    for idx, node in enumerate(nodes):
        node["cluster_center_text_embedding"] = node.get("text_embedding")
        node["text_embedding"] = embeddings[offset + idx].cpu()
        node["text_embedding_source"] = "clip_encoded_summary_text"
        node["summary_clip_model"] = "clip-vit-large-patch14-336"


def recompute_primary_global_edges(layer2_nodes, global_nodes):
    layer2_by_id = {int(node["global_node_id"]): node for node in layer2_nodes}
    global_by_id = {int(node["global_node_id"]): node for node in global_nodes}
    edge_parts = []
    for proto in global_nodes:
        member_ids = [int(x) for x in proto.get("member_layer2_abstract_node_ids", [])]
        src_ids = [mid for mid in member_ids if mid in layer2_by_id]
        if not src_ids:
            continue
        src = torch.tensor(src_ids, dtype=torch.long)
        dst = torch.full_like(src, int(proto["global_node_id"]))
        proto_emb = F.normalize(global_by_id[int(proto["global_node_id"])]["text_embedding"].float(), dim=0)
        member_emb = F.normalize(torch.stack([layer2_by_id[mid]["text_embedding"].float() for mid in src_ids]), dim=-1)
        conf = (member_emb @ proto_emb).cpu()
        edge_parts.append({"edge_index": torch.stack([src, dst], dim=1), "confidence": conf})
        edge_parts.append({"edge_index": torch.stack([dst, src], dim=1), "confidence": conf})
    return edge_parts


def recompute_topm_edges(layer2_nodes, global_nodes, top_m):
    layer2_emb = F.normalize(torch.stack([node["text_embedding"].float() for node in layer2_nodes]), dim=-1)
    global_emb = F.normalize(torch.stack([node["text_embedding"].float() for node in global_nodes]), dim=-1)
    sims = layer2_emb @ global_emb.T
    top_k = min(top_m, len(global_nodes))
    top_conf, top_idx = torch.topk(sims, k=top_k, dim=1)
    src_nodes, dst_nodes, confs, ranks = [], [], [], []
    for layer2_pos, layer2_node in enumerate(layer2_nodes):
        for rank in range(top_k):
            proto = global_nodes[int(top_idx[layer2_pos, rank].item())]
            src_nodes.append(int(layer2_node["global_node_id"]))
            dst_nodes.append(int(proto["global_node_id"]))
            confs.append(float(top_conf[layer2_pos, rank].item()))
            ranks.append(rank + 1)
    src = torch.tensor(src_nodes, dtype=torch.long)
    dst = torch.tensor(dst_nodes, dtype=torch.long)
    confidence = torch.tensor(confs, dtype=torch.float32)
    rank_tensor = torch.tensor(ranks, dtype=torch.long)
    return [
        {
            "edge_index": torch.stack([src, dst], dim=1),
            "confidence": confidence,
            "rank": rank_tensor,
            "edge_type": "layer2-to-global-semantic-topm",
        },
        {
            "edge_index": torch.stack([dst, src], dim=1),
            "confidence": confidence,
            "rank": rank_tensor,
            "edge_type": "global-to-layer2-semantic-topm",
        },
    ]


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    model_path = Path(args.model_path)
    peak = estimate_peak_gb(input_root, model_path)
    print(json.dumps({"stage": "estimate", "estimated_peak_gb": round(float(peak), 3), "memory_limit_gb": args.memory_limit_gb}), flush=True)
    if peak > args.memory_limit_gb:
        raise RuntimeError(f"estimated peak {peak:.2f}GB exceeds limit {args.memory_limit_gb:.2f}GB")
    if args.estimate_only:
        return

    copy_tree(input_root, output_root)
    layer1 = torch.load(output_root / "layer1_abstract_nodes.pt", map_location="cpu", weights_only=False)
    layer2 = torch.load(output_root / "layer2_abstract_nodes.pt", map_location="cpu", weights_only=False)
    global_nodes = torch.load(output_root / "global_semantic_prototype_nodes.pt", map_location="cpu", weights_only=False)
    all_nodes = layer1 + layer2 + global_nodes
    texts = [node.get("summary_text", "") for node in all_nodes]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
    validate_text_tokenizer(processor)
    model = CLIPModel.from_pretrained(str(model_path), local_files_only=True).eval().to(device)
    embeddings = encode_texts(model, processor, texts, args.batch_size, device)
    if embeddings.shape[0] > 1 and torch.equal(embeddings[0], embeddings[1]):
        raise RuntimeError("Encoded summary embeddings collapsed; aborting.")

    replace_embeddings(layer1, embeddings, 0)
    replace_embeddings(layer2, embeddings, len(layer1))
    replace_embeddings(global_nodes, embeddings, len(layer1) + len(layer2))

    torch.save(layer1, output_root / "layer1_abstract_nodes.pt")
    torch.save(layer2, output_root / "layer2_abstract_nodes.pt")
    torch.save(global_nodes, output_root / "global_semantic_prototype_nodes.pt")

    primary_edges = recompute_primary_global_edges(layer2, global_nodes)
    torch.save(primary_edges, output_root / "abstract_global_edges.pt")
    meta_path = output_root / "kmeans_global_synthetic_graph_meta.json"
    meta = json.load(meta_path.open("r", encoding="utf-8"))
    top_m = int(meta.get("global_top_m", 3))
    torch.save(recompute_topm_edges(layer2, global_nodes, top_m), output_root / "abstract_global_topm_edges.pt")
    meta["summary_text_reencoded"] = {
        "enabled": True,
        "model_path": str(model_path),
        "text_embedding_source": "CLIP text encoder over rewritten summary_text",
        "previous_cluster_center_saved_as": "cluster_center_text_embedding",
        "global_edge_confidence_recomputed": True,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"stage": "done", "output_root": str(output_root), "encoded_summaries": len(texts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
