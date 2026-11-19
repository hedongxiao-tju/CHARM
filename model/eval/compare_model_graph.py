import argparse
import json
import os
import subprocess
import sys


DEFAULT_MODEL_BASE = "models/vicuna-7b-v1.5"
DEFAULT_DATA_ROOT = "data/ele-fashion"


def lookup_ground_truth(data_root, asin):
    text_path = os.path.join(data_root, "ele-fashion-raw-text.jsonl")
    with open(text_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["asin"] == asin:
                return record["raw_text"]
    return None


def run_model(model_path, model_base, data_root, asin, node_id, query, device, graph_hops, graph_sample_size):
    cmd = [
        sys.executable,
        "-m",
        "model.eval.run_model_graph",
        "--model-path",
        model_path,
        "--data-root",
        data_root,
        "--graph-only",
        "--graph-hops",
        str(graph_hops),
        "--graph-sample-size",
        str(graph_sample_size),
        "--query",
        query,
        "--device",
        device,
    ]
    if model_base:
        cmd.extend(["--model-base", model_base])
    if asin is not None:
        cmd.extend(["--asin", asin])
    if node_id is not None:
        cmd.extend(["--node-id", str(node_id)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Model run failed for {model_path}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-model-path", type=str, required=True)
    parser.add_argument("--after-model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--asin", type=str, default=None)
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-sample-size", type=int, default=10)
    args = parser.parse_args()

    if args.asin is None and args.node_id is None:
        raise ValueError("Provide either --asin or --node-id.")

    ground_truth = lookup_ground_truth(args.data_root, args.asin) if args.asin is not None else None

    before_output = run_model(
        model_path=args.before_model_path,
        model_base=args.model_base,
        data_root=args.data_root,
        asin=args.asin,
        node_id=args.node_id,
        query=args.query,
        device=args.device,
        graph_hops=args.graph_hops,
        graph_sample_size=args.graph_sample_size,
    )
    after_output = run_model(
        model_path=args.after_model_path,
        model_base=args.model_base,
        data_root=args.data_root,
        asin=args.asin,
        node_id=args.node_id,
        query=args.query,
        device=args.device,
        graph_hops=args.graph_hops,
        graph_sample_size=args.graph_sample_size,
    )

    print("=" * 80)
    print(f"ASIN: {args.asin}" if args.asin is not None else f"Node ID: {args.node_id}")
    print(f"Query: {args.query}")
    if ground_truth is not None:
        print(f"Ground Truth: {ground_truth}")
    print("-" * 80)
    print(f"[Before] {args.before_model_path}")
    print(before_output)
    print("-" * 80)
    print(f"[After]  {args.after_model_path}")
    print(after_output)
    print("=" * 80)


if __name__ == "__main__":
    main()
