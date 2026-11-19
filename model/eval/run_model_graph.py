import argparse
import html
import io
import json
import os
import random
import tarfile
import re

import requests
import torch
from PIL import Image
from io import BytesIO

from model.constants import (
    DEFAULT_GRAPH_PAD_ID,
    DEFAULT_GRAPH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
)
from model.conversation import conv_templates
from model.model.builder import load_pretrained_model
from model.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_graph_image_token,
)
from model.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def load_edge_list(edge_path, num_nodes):
    edge_pairs = torch.load(edge_path, map_location="cpu", weights_only=False)
    edge_list = [[] for _ in range(num_nodes)]
    for src, dst in edge_pairs:
        src = int(src)
        dst = int(dst)
        edge_list[src].append(dst)
        edge_list[dst].append(src)
    return edge_list


def bfs_serialize(edge_list, node_idx, k_hop, sample_size, pad_val=DEFAULT_GRAPH_PAD_ID):
    neighbors = [[int(node_idx)]]
    for _ in range(k_hop):
        current_hop = []
        for parent in neighbors[-1]:
            if parent == pad_val:
                current_hop.extend([pad_val] * sample_size)
                continue
            node_neighbors = edge_list[parent]
            if len(node_neighbors) > sample_size:
                sampled_neighbors = random.sample(node_neighbors, sample_size)
            else:
                sampled_neighbors = node_neighbors + [pad_val] * (sample_size - len(node_neighbors))
            current_hop.extend(sampled_neighbors)
        neighbors.append(current_hop)
    return [node for hop in neighbors for node in hop]


def load_node_mapping(path):
    if not os.path.exists(path):
        return {}
    raw_mapping = torch.load(path, map_location="cpu", weights_only=False)
    mapping = {}
    if isinstance(raw_mapping, dict):
        for key, value in raw_mapping.items():
            if isinstance(value, dict):
                value = value.get("node_id", value.get("id", value.get("idx")))
            if value is not None:
                mapping[str(key)] = int(value)
    elif isinstance(raw_mapping, (list, tuple)):
        for idx, item in enumerate(raw_mapping):
            if isinstance(item, str):
                mapping[item] = idx
            elif isinstance(item, dict):
                asin = item.get("asin") or item.get("id")
                node_id = item.get("node_id", item.get("idx", idx))
                if asin is not None:
                    mapping[str(asin)] = int(node_id)
    return mapping


def resolve_node_id(args):
    if args.node_id is not None:
        return args.node_id
    if args.asin is None:
        raise ValueError("Provide either --node-id or --asin.")
    mapping = load_node_mapping(os.path.join(args.data_root, "node_mapping.pt"))
    if args.asin in mapping:
        return mapping[args.asin]

    text_path = os.path.join(args.data_root, "ele-fashion-raw-text.jsonl")
    with open(text_path, "r") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            if record["asin"] == args.asin:
                return idx
    raise ValueError(f"Unable to resolve node id for ASIN {args.asin}")


def resolve_node_id_pair(args):
    mapping = load_node_mapping(os.path.join(args.data_root, "node_mapping.pt"))

    if args.node_u is not None:
        node_u = args.node_u
    elif args.asin_u is not None and args.asin_u in mapping:
        node_u = mapping[args.asin_u]
    else:
        raise ValueError("Provide --node-u or --asin-u for link prediction.")

    if args.node_v is not None:
        node_v = args.node_v
    elif args.asin_v is not None and args.asin_v in mapping:
        node_v = mapping[args.asin_v]
    else:
        raise ValueError("Provide --node-v or --asin-v for link prediction.")

    return int(node_u), int(node_v)


def resolve_image(args):
    if args.graph_only:
        return None
    if args.image_file is not None:
        return load_image(args.image_file)
    if args.asin is None:
        raise ValueError("Provide --image-file or --asin.")

    if args.image_folder is not None:
        candidate = os.path.join(args.image_folder, f"{args.asin}.jpg")
        if os.path.exists(candidate):
            return load_image(candidate)

    archive_path = os.path.join(args.data_root, "ele-fashion-images.tar")
    member_name = f"ele-fashion-images/{args.asin}.jpg"
    with tarfile.open(archive_path, "r") as tar:
        member = tar.getmember(member_name)
        with tar.extractfile(member) as fp:
            return Image.open(io.BytesIO(fp.read())).convert("RGB")


def build_graph_inputs(args, node_id):
    feat_path = os.path.join(args.data_root, "clip_feat.pt")
    edge_path = os.path.join(args.data_root, "nc_edges-nodeid.pt")

    graph_features = torch.load(feat_path, map_location="cpu", weights_only=False)
    if isinstance(graph_features, dict):
        graph_features = graph_features["features"]

    edge_list = load_edge_list(edge_path, graph_features.shape[0])
    graph_ids = torch.tensor(
        bfs_serialize(
            edge_list=edge_list,
            node_idx=node_id,
            k_hop=args.graph_hops,
            sample_size=args.graph_sample_size,
        ),
        dtype=torch.long,
    )
    graph_node_mask = graph_ids.ne(DEFAULT_GRAPH_PAD_ID)
    graph_emb = torch.zeros((graph_ids.shape[0], graph_features.shape[1]), dtype=torch.float32)
    graph_emb[graph_node_mask] = graph_features[graph_ids[graph_node_mask]].to(torch.float32)
    return graph_ids.unsqueeze(0), graph_emb.unsqueeze(0), graph_node_mask.unsqueeze(0)


def build_graph_inputs_pair(args, node_u, node_v):
    feat_path = os.path.join(args.data_root, "clip_feat.pt")
    edge_path = os.path.join(args.data_root, "nc_edges-nodeid.pt")

    graph_features = torch.load(feat_path, map_location="cpu", weights_only=False)
    if isinstance(graph_features, dict):
        graph_features = graph_features["features"]

    edge_list = load_edge_list(edge_path, graph_features.shape[0])
    graph_u = torch.tensor(
        bfs_serialize(edge_list=edge_list, node_idx=node_u, k_hop=args.graph_hops, sample_size=args.graph_sample_size),
        dtype=torch.long,
    )
    graph_v = torch.tensor(
        bfs_serialize(edge_list=edge_list, node_idx=node_v, k_hop=args.graph_hops, sample_size=args.graph_sample_size),
        dtype=torch.long,
    )

    graphs = torch.stack([graph_u, graph_v], dim=0)
    graph_node_mask = graphs.ne(DEFAULT_GRAPH_PAD_ID)
    graph_emb = torch.zeros((2, graphs.shape[1], graph_features.shape[1]), dtype=torch.float32)
    graph_emb[0, graph_node_mask[0]] = graph_features[graph_u[graph_node_mask[0]]].to(torch.float32)
    graph_emb[1, graph_node_mask[1]] = graph_features[graph_v[graph_node_mask[1]]].to(torch.float32)
    return graphs.unsqueeze(0), graph_emb.unsqueeze(0), graph_node_mask.unsqueeze(0)


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model_device = next(model.parameters()).device

    image = resolve_image(args)
    image_size = image.size if image is not None else None
    image_tensor = None
    if image is not None:
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [x.to(model_device, dtype=torch.float16) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(model_device, dtype=torch.float16)

    if args.task == "lp":
        node_u, node_v = resolve_node_id_pair(args)
        graphs, graph_emb, graph_node_mask = build_graph_inputs_pair(args, node_u, node_v)
    else:
        node_id = resolve_node_id(args)
        graphs, graph_emb, graph_node_mask = build_graph_inputs(args, node_id)
    graphs = graphs.to(model_device)
    graph_emb = graph_emb.to(model_device, dtype=model.dtype)
    graph_node_mask = graph_node_mask.to(model_device)

    qs = args.query
    graph_token = DEFAULT_GRAPH_TOKEN
    if args.task == "lp":
        if args.query is None:
            qs = (
                "Given two node-centered subgraphs: <graph> and <graph>, "
                "predict whether the two center nodes should be connected. "
                "Answer with yes or no only."
            )
    if args.graph_only:
        if args.task == "lp":
            if "<graph>" in qs:
                pass
            else:
                qs = f"{graph_token}\n{graph_token}\n" + qs
        else:
            qs = graph_token + "\n" + qs
    else:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token, qs)
        else:
            qs = image_token + "\n" + graph_token + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None:
        conv_mode = args.conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_graph_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(model_device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size] if image_size is not None else None,
            graphs=graphs,
            graph_emb=graph_emb,
            graph_node_mask=graph_node_mask,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--task", type=str, default="nc", choices=["nc", "lp"])
    parser.add_argument("--asin", type=str, default=None)
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--asin-u", type=str, default=None)
    parser.add_argument("--asin-v", type=str, default=None)
    parser.add_argument("--node-u", type=int, default=None)
    parser.add_argument("--node-v", type=int, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-sample-size", type=int, default=10)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
