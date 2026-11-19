#!/usr/bin/env python3
import argparse
import html
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


DATASETS = {
    "Grocery": {"csv": "Grocery.csv", "text_col": "text"},
    "Toys": {"csv": "Toys.csv", "text_col": "text"},
    "Movies": {"csv": "Movies.csv", "text_col": "text"},
    "Reddit-S": {"csv": "RedditS.csv", "text_col": "caption"},
    "Health": {"jsonl": "health-raw-text.jsonl", "text_col": "raw_text"},
    "Beauty": {"jsonl": "beauty-raw-text.jsonl", "text_col": "raw_text"},
    "Arts": {"jsonl": "arts-raw-text.jsonl", "text_col": "raw_text"},
    "CD": {"jsonl": "cd-raw-text.jsonl", "text_col": "raw_text"},
}

EXTRA_STOPWORDS = {
    "pack", "count", "ounce", "ounces", "oz", "lb", "lbs", "inch", "inches",
    "new", "used", "great", "best", "set", "piece", "pieces", "amazon",
    "product", "item", "items", "features", "includes", "featuring", "quality",
    "movie", "film", "dvd", "blu", "ray", "video", "edition", "series",
    "reddit", "post", "image", "photo", "picture", "comments",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite MAGB synthetic abstract node summaries as concise semantic descriptions.")
    parser.add_argument("--data-root", default="data/MAGB")
    parser.add_argument("--input-root", default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3")
    parser.add_argument("--output-root", default="data/MAGB/kmeans_hierarchical_global_synthetic_textfixed_g24_topm3_semantic")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--top-texts", type=int, default=80)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--memory-limit-gb", type=float, default=20.0)
    parser.add_argument("--estimate-only", action="store_true")
    return parser.parse_args()


def clean_text(value, max_chars=500):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^A-Za-z0-9&%+./' -]+", " ", text)
    text = " ".join(text.split())
    return text[:max_chars]


def load_dataset_context(data_root: Path, dataset: str):
    spec = DATASETS[dataset]
    root = data_root / dataset
    if "jsonl" in spec:
        import json as _json
        texts = []
        with open(root / spec["jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                rec = _json.loads(line.strip())
                raw = rec.get(spec["text_col"], "")
                text = clean_text(raw, 800)
                texts.append(text)
        return {"texts": texts}
    header = pd.read_csv(root / spec["csv"], nrows=0).columns.tolist()
    usecols = ["id", spec["text_col"]]
    for optional in ["title", "description"]:
        if optional in header:
            usecols.append(optional)
    df = pd.read_csv(root / spec["csv"], usecols=list(dict.fromkeys(usecols))).sort_values("id").reset_index(drop=True)
    texts = []
    for row in df.itertuples(index=False):
        title = clean_text(getattr(row, "title", ""), 160)
        description = clean_text(getattr(row, "description", ""), 360)
        body = clean_text(getattr(row, spec["text_col"], ""), 360)
        text = " ".join(x for x in [title, description, body] if x)
        texts.append(text)
    return {"texts": texts}


def estimate_peak_gb(data_root: Path, input_root: Path, datasets):
    size = 0
    for path in input_root.glob("*.pt"):
        size += path.stat().st_size
    for dataset in datasets:
        spec = DATASETS[dataset]
        fname = spec.get("jsonl", spec.get("csv"))
        fpath = data_root / dataset / fname
        if fpath.exists():
            size += fpath.stat().st_size
    return (size * 2.5 + 1024**3) / (1024**3)


def short_join(values, max_items=4):
    out = []
    seen = set()
    for value in values:
        value = clean_text(value, 80)
        key = value.lower()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def extract_terms(texts, top_terms=8):
    docs = [t for t in texts if t]
    if not docs:
        return []
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9&%+'.-]{2,}\b",
            max_features=1500,
        )
        x = vectorizer.fit_transform(docs)
    except ValueError:
        return []
    scores = x.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
    selected = []
    seen_words = set()
    for term, _ in ranked:
        words = [w for w in term.lower().split() if w not in EXTRA_STOPWORDS]
        if not words:
            continue
        if any(word in seen_words for word in words) and len(words) == 1:
            continue
        phrase = " ".join(words)
        if len(phrase) < 3:
            continue
        selected.append(phrase)
        seen_words.update(words)
        if len(selected) >= top_terms:
            break
    return selected


def make_semantic_label(dataset, terms):
    if terms:
        label = " / ".join(term.title() for term in terms[:2])
        if dataset == "Reddit-S":
            return f"{label} visual discussion"
        return label
    return f"{dataset} semantic group"


def summarize_layer1_node(node, ctx, top_texts, top_terms):
    ids = node["child_node_ids"]
    texts = [ctx["texts"][int(i)] for i in ids if ctx["texts"][int(i)]]
    texts = texts[:top_texts]
    terms = extract_terms(texts, top_terms)
    semantic_label = make_semantic_label(node["dataset"], terms)
    term_text = ", ".join(terms[:5]) if terms else "shared textual and visual attributes"
    if node["dataset"] == "Reddit-S":
        summary = (
            f"{semantic_label}. A Reddit-S semantic group centered on {term_text}. "
            f"The posts share related visual scenes or discussion contexts without using subreddit or label names."
        )
    else:
        summary = (
            f"{semantic_label}. A {node['dataset']} semantic group about {term_text}. "
            f"The nodes represent a coherent product concept derived only from titles and descriptions."
        )
    return semantic_label, summary


def summarize_parent_node(node, member_nodes, level_name):
    labels = [m.get("semantic_label", "") for m in member_nodes if m.get("semantic_label")]
    terms = extract_terms([m.get("summary_text", "") for m in member_nodes], 10)
    common = short_join([name for name, _ in Counter(labels).most_common(5)], 5)
    if common:
        semantic_label = " / ".join(common[:2])
    elif terms:
        semantic_label = " / ".join(term.title() for term in terms[:2])
    else:
        semantic_label = f"{node['dataset']} semantic topic"
    term_text = ", ".join(terms[:6]) if terms else ", ".join(common[:4])
    dataset_text = node["dataset"] if node["dataset"] != "global" else "cross-dataset"
    summary = (
        f"{semantic_label}. A {dataset_text} {level_name} that consolidates related abstract groups around {term_text}. "
        f"It provides a higher-level semantic anchor for zero-shot reasoning across its covered nodes."
    )
    return semantic_label, summary


def copy_tree(input_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    for path in input_root.iterdir():
        dst = output_root / path.name
        if path.is_file():
            shutil.copy2(path, dst)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    meta_path = input_root / "kmeans_global_synthetic_graph_meta.json"
    input_meta = json.load(meta_path.open("r", encoding="utf-8")) if meta_path.exists() else {}
    selected_datasets = args.datasets or input_meta.get("datasets") or list(DATASETS.keys())
    selected_datasets = [dataset for dataset in selected_datasets if dataset in DATASETS]
    peak = estimate_peak_gb(data_root, input_root, selected_datasets)
    print(json.dumps({"stage": "estimate", "estimated_peak_gb": round(float(peak), 3), "memory_limit_gb": args.memory_limit_gb}), flush=True)
    if peak > args.memory_limit_gb:
        raise RuntimeError(f"estimated peak {peak:.2f}GB exceeds limit {args.memory_limit_gb:.2f}GB")
    if args.estimate_only:
        return

    copy_tree(input_root, output_root)
    contexts = {dataset: load_dataset_context(data_root, dataset) for dataset in selected_datasets}
    layer1 = torch.load(output_root / "layer1_abstract_nodes.pt", map_location="cpu", weights_only=False)
    layer2 = torch.load(output_root / "layer2_abstract_nodes.pt", map_location="cpu", weights_only=False)
    global_nodes = torch.load(output_root / "global_semantic_prototype_nodes.pt", map_location="cpu", weights_only=False)

    for node in layer1:
        node["source_summary_text"] = node.get("summary_text", "")
        semantic_label, summary = summarize_layer1_node(node, contexts[node["dataset"]], args.top_texts, args.top_terms)
        node["semantic_label"] = semantic_label
        node["summary_text"] = summary
        node["summary_method"] = "cluster_text_tfidf_semantic_rewrite"

    layer1_by_id = {int(node["global_node_id"]): node for node in layer1}
    for node in layer2:
        node["source_summary_text"] = node.get("summary_text", "")
        members = [layer1_by_id[int(i)] for i in node["child_abstract_node_ids"] if int(i) in layer1_by_id]
        semantic_label, summary = summarize_parent_node(node, members, "high-level topic")
        node["semantic_label"] = semantic_label
        node["summary_text"] = summary
        node["summary_method"] = "child_abstract_semantic_rewrite"

    layer2_by_id = {int(node["global_node_id"]): node for node in layer2}
    for node in global_nodes:
        node["source_summary_text"] = node.get("summary_text", "")
        member_ids = node.get("member_layer2_abstract_node_ids", [])
        members = [layer2_by_id[int(i)] for i in member_ids if int(i) in layer2_by_id]
        semantic_label, summary = summarize_parent_node(node, members, "global semantic prototype")
        node["semantic_label"] = semantic_label
        node["summary_text"] = summary
        node["summary_method"] = "child_abstract_semantic_rewrite"

    torch.save(layer1, output_root / "layer1_abstract_nodes.pt")
    torch.save(layer2, output_root / "layer2_abstract_nodes.pt")
    torch.save(global_nodes, output_root / "global_semantic_prototype_nodes.pt")
    meta_path = output_root / "kmeans_global_synthetic_graph_meta.json"
    meta = json.load(meta_path.open("r", encoding="utf-8"))
    meta["summary_rewrite"] = {
        "enabled": True,
        "method": "semantic rewrite from clean clustered titles/descriptions and extracted topic terms; labels/categories/subreddit names are not used",
        "source_summary_text_preserved": True,
        "output_root": str(output_root),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    preview = {
        "stage": "done",
        "output_root": str(output_root),
        "layer1": len(layer1),
        "layer2": len(layer2),
        "global": len(global_nodes),
        "samples": [
            {"id": node["id"], "semantic_label": node["semantic_label"], "summary_text": node["summary_text"]}
            for node in layer1[:3]
        ],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
