"""Command line interface for CHARM data preparation, training and evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

from .common import (REPO, DEFAULT_CLIP, DEFAULT_DATA, DEFAULT_GRAPH, DEFAULT_INIT,
                     DEFAULT_MODEL, fail_missing, python, require_fresh, run,
                     validate_domains)

SCRIPTS = REPO / 'scripts'


def context_options(p, *, sources=True):
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--synthetic-root', type=Path, default=DEFAULT_GRAPH)
    if sources:
        p.add_argument('--sources', nargs=2, default=['Grocery', 'Movies'])
        p.add_argument('--target', default='Toys')


def runtime_options(p):
    p.add_argument('--gpu', default='0', help='CUDA device index')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running them')


def check_data(args):
    validate_domains(args.sources, args.target)
    domains = [*args.sources, args.target]
    required = [args.synthetic_root / 'dataset_offsets.json',
                args.synthetic_root / 'kmeans_global_synthetic_graph_meta.json',
                args.synthetic_root / 'modality_complement_bridge_edges.pt']
    for domain in domains:
        root = args.data_root / domain
        required += [root / 'clip_feat.pt', root / 'graph_label_config.json',
                     root / 'ppr.pt', root / 'lp_zeroshot_split.pt']
    fail_missing(required)
    offsets = json.loads((args.synthetic_root / 'dataset_offsets.json').read_text())
    missing = [name for name in domains if name not in offsets]
    if missing:
        raise ValueError(f'Synthetic graph is missing domains: {missing}')
    import torch
    print('Inputs: OK')
    print(f'Data root: {args.data_root}\nSynthetic graph: {args.synthetic_root}')
    for domain in domains:
        split = torch.load(args.data_root / domain / 'lp_zeroshot_split.pt',
                           map_location='cpu', weights_only=False)
        rows = split['train']
        if not isinstance(rows, list):
            print(f'{domain}: LP training split uses a non-list format; inspect before training')
            continue
        subset = rows[:args.max_lp_samples] if args.max_lp_samples > 0 else rows
        labels = [int(row['label']) for row in subset]
        print(f'{domain}: train={len(rows)}, selected={len(subset)}, '
              f'positive={labels.count(1)}, negative={labels.count(0)}, test={len(split["test"])}')
        if domain in args.sources and len(set(labels)) < 2:
            raise ValueError(f'{domain}: selected LP training examples contain only one class')
    print('LP graph and PPR cache:', args.synthetic_root)


def prepare_graph(args):
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Dataset names must be unique')
    if not args.datasets:
        raise ValueError('Specify at least one dataset')
    for ds in args.datasets:
        fail_missing([args.data_root / ds / name for name in
                      ('clip_feat.pt', 'features_clip_text.pt', 'features_clip_image.pt', 'ppr.pt')])
    fail_missing([args.clip_model])
    stages = [args.output_root / name for name in ('01_hierarchy', '02_summaries', '03_cliptext', '04_bridges')]
    for path in stages:
        require_fresh(path)
    dsargs = ['--datasets', *args.datasets]
    commands = [
        python(SCRIPTS / 'build_magb_kmeans_global_synthetic_graph.py', '--data-root', args.data_root,
               '--output-root', stages[0], *dsargs, '--no-mm-sc', '--global-prototypes', '24', '--global-top-m', '3'),
        python(SCRIPTS / 'rewrite_magb_abstract_summaries.py', '--data-root', args.data_root,
               '--input-root', stages[0], '--output-root', stages[1], *dsargs),
        python(SCRIPTS / 'reencode_magb_abstract_summary_text.py', '--input-root', stages[1],
               '--output-root', stages[2], '--model-path', args.clip_model, '--device', 'cuda:0'),
        python(SCRIPTS / 'build_magb_modality_complement_bridges.py', '--data-root', args.data_root,
               '--input-root', stages[2], '--output-root', stages[3], *dsargs),
    ]
    for command in commands:
        run(command, dry_run=args.dry_run, gpu=args.gpu)
    print(f'Final synthetic graph: {stages[-1]}')


def validate_training_splits(data_root, sources, limit):
    """Match the legacy dataset's selection and refuse a one-class LP subset."""
    import torch
    for domain in sources:
        split = torch.load(data_root / domain / 'lp_zeroshot_split.pt',
                           map_location='cpu', weights_only=False)
        rows = split['train']
        if isinstance(rows, list):
            chosen = rows[:limit] if limit > 0 else rows
            labels = {int(row['label']) for row in chosen}
        elif isinstance(rows, dict) and 'pos_edges' in rows and 'neg_edges' in rows:
            # The current legacy dataset concatenates positives before negatives.
            positives = len(rows['pos_edges'])
            negatives = len(rows['neg_edges'])
            selected_count = min(positives + negatives, limit) if limit > 0 else positives + negatives
            labels = ({1} if positives and selected_count else set())
            if selected_count > positives and negatives:
                labels.add(0)
        else:
            raise ValueError(f'{domain}: unsupported LP training split')
        if labels != {0, 1}:
            raise ValueError(f'{domain}: chosen LP examples do not contain both classes; '
                             'increase --max-lp-samples or inspect the split ordering')


def prepare_splits(args):
    fail_missing([args.synthetic_root / 'dataset_offsets.json', args.synthetic_root / 'original_graph_edges.pt'])
    for ds in args.datasets:
        if Path(ds).name != ds or ds in ('.', '..'):
            raise ValueError(f'Invalid dataset name: {ds}')
        fail_missing([args.data_root / ds / 'ppr.pt'])
    outputs = [args.data_root / ds / 'lp_zeroshot_split.pt' for ds in args.datasets]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.replace_existing:
        raise FileExistsError('LP splits already exist; use --replace-existing to back them up and rebuild:\n  '
                              + '\n  '.join(map(str, existing)))
    if existing and not args.dry_run:
        backup = REPO / 'charm_backups' / ('lp_splits_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, backup / (path.parent.name + '.pt'))
        print(f'Existing splits backed up to {backup}', flush=True)
    run(python(REPO / 'prepare_lp_splits.py', '--magb-root', args.data_root,
               '--syn-root', args.synthetic_root, '--datasets', *args.datasets,
               '--seed', args.seed, '--hard-neg-ratio', args.hard_neg_ratio,
               '--test-hard-neg-ratio', args.test_hard_neg_ratio), dry_run=args.dry_run)


def train(args):
    validate_domains(args.sources, args.target)
    fail_missing([args.model / 'config.json', args.init_adapter, args.synthetic_root / 'dataset_offsets.json'])
    for domain in args.sources:
        fail_missing([args.data_root / domain / f for f in
                      ('clip_feat.pt', 'graph_label_config.json', 'lp_zeroshot_split.pt')])
    validate_training_splits(args.data_root, args.sources, args.max_lp_samples)
    require_fresh(args.output_dir)
    model_config = json.loads((args.model / 'config.json').read_text())
    plain_vicuna = model_config.get('model_type') == 'llama'
    vision_args = []
    if plain_vicuna:
        fail_missing([args.clip_model / 'config.json'])
        vision_args = ['--vision_tower', str(args.clip_model), '--mm_projector_type', 'mlp2x_gelu',
                       '--mm_vision_select_layer', '-2', '--mm_use_im_patch_token', 'False']
    roots = [str(args.data_root / ds) for ds in args.sources] * 2
    cmd = python(REPO / 'charm_train.py', '--model_name_or_path', args.model, '--version', 'v1',
        '--graph_data_roots', ','.join(roots), '--graph_feat_paths', ','.join(['clip_feat.pt'] * 4),
        '--graph_label_configs', ','.join(['graph_label_config.json'] * 4),
        '--graph_synthetic_root', args.synthetic_root, '--graph_label_supervision', 'True', '--graph_only', 'True',
        '--graph_zero_shot_serializer', 'True', '--graph_include_label_descriptions', 'True',
        '--graph_answer_with_description', 'False', '--graph_label_prompt_style', 'default',
        '--graph_semantic_coord_tokens', 'True', '--graph_include_abstract_nodes', 'True',
        '--graph_include_synthetic_soft_tokens', 'True', '--graph_use_soft_tokens', 'True',
        '--graph_hops', '2', '--graph_sample_size', '4', '--graph_real_neighbor_topk', '8',
        '--graph_real_cross_topk', '4', '--graph_cache_paths', ',,,', '--graph_feat_cache_paths', ',,,',
        '--freeze_backbone', 'True', '--tune_mm_mlp_adapter', 'True', '--bits', '16', '--bf16', 'True',
        '--num_train_epochs', args.epochs, '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', args.grad_accum, '--learning_rate', args.learning_rate,
        '--weight_decay', '0.0', '--warmup_ratio', '0.03', '--lr_scheduler_type', 'cosine',
        '--logging_steps', '50', '--save_steps', '99999', '--save_total_limit', '1', '--report_to', 'none',
        '--dataloader_num_workers', '0', '--model_max_length', '2048',
        '--pretrain_mm_mlp_adapter', args.init_adapter, '--output_dir', args.output_dir, *vision_args)
    run(cmd, dry_run=args.dry_run, gpu=args.gpu,
        env_extra={'CHARM_TASKS': 'nc,nc,lp,lp', 'CHARM_MAX_LP_SAMPLES': str(args.max_lp_samples),
                   'CHARM_INITIALIZE_ADAPTER': '1'})


def eval_nc(args):
    fail_missing([args.checkpoint / 'config.json', args.checkpoint / 'mm_projector.bin',
                  args.model / 'config.json', args.data_root / args.target / 'test_mask.pt',
                  args.synthetic_root / 'dataset_offsets.json'])
    require_fresh(args.output_dir)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = python('-m', 'model.eval.eval_magb_graph_zeroshot', '--model-path', args.checkpoint,
        '--model-base', args.model, '--max-samples', args.max_samples,
        '--graph-sample-size', '4', '--graph-hops', '2', '--graph-real-node-serializer',
        '--graph-real-neighbor-topk', '8', '--graph-real-cross-topk', '4',
        '--score-mode', 'option_logits', '--calibrate-label-prior',
        '--data-root', args.data_root / args.target,
        '--synthetic-root', args.synthetic_root, '--answers-file', args.output_dir / 'predictions.jsonl')
    run(cmd, dry_run=args.dry_run, gpu=args.gpu)


def eval_lp(args):
    fail_missing([args.checkpoint / 'config.json', args.checkpoint / 'mm_projector.bin',
                  args.model / 'config.json', args.data_root / args.target / 'lp_zeroshot_split.pt',
                  args.synthetic_root / 'dataset_offsets.json'])
    require_fresh(args.output_dir)
    validate_domains(args.sources, args.target)
    if args.max_samples != -1:
        raise ValueError('The legacy LP evaluator does not support --max-samples; use -1')
    print('LP evaluation: fixed probability threshold 0.5.', flush=True)
    cmd = python(REPO / 'lp_zeroshot.py', '--model-path', args.checkpoint, '--model-base', args.model,
                 '--source-domains', *args.sources, '--target-domain', args.target,
                 '--magb-root', args.data_root, '--synthetic-root', args.synthetic_root,
                 '--output-dir', args.output_dir, '--eval-only', '--load-checkpoint', args.checkpoint,
                 '--eval-mode', args.eval_mode)
    run(cmd, dry_run=args.dry_run, gpu=args.gpu)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    q = subs.add_parser('check', help='Inspect data and LP class counts without changing files')
    context_options(q); q.add_argument('--max-lp-samples', type=int, default=5000); q.set_defaults(func=check_data)
    q = subs.add_parser('prepare-graph', help='Build hierarchy, summaries, CLIP embeddings and bridges')
    q.add_argument('--data-root', type=Path, default=DEFAULT_DATA); runtime_options(q)
    q.add_argument('--datasets', nargs='+', default=['Grocery', 'Movies', 'Toys'])
    q.add_argument('--clip-model', type=Path, default=DEFAULT_CLIP)
    q.add_argument('--output-root', type=Path, required=True); q.set_defaults(func=prepare_graph)
    q = subs.add_parser('prepare-splits', help='Build LP splits; existing splits are protected')
    context_options(q, sources=False); q.add_argument('--datasets', nargs='+', default=['Grocery', 'Movies', 'Toys'])
    q.add_argument('--seed', type=int, default=42); q.add_argument('--hard-neg-ratio', type=float, default=0.5)
    q.add_argument('--test-hard-neg-ratio', type=float, default=0.0)
    q.add_argument('--replace-existing', action='store_true'); q.add_argument('--dry-run', action='store_true'); q.set_defaults(func=prepare_splits)
    q = subs.add_parser('train', help='Joint NC and LP training on two source domains')
    context_options(q); runtime_options(q); q.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    q.add_argument('--init-adapter', type=Path, default=DEFAULT_INIT)
    q.add_argument('--clip-model', type=Path, default=DEFAULT_CLIP, help='Local vision encoder directory for base-model training')
    q.add_argument('--output-dir', type=Path, required=True); q.add_argument('--epochs', type=float, default=1)
    q.add_argument('--grad-accum', type=int, default=8); q.add_argument('--learning-rate', type=float, default=2e-4)
    q.add_argument('--max-lp-samples', type=int, default=5000); q.set_defaults(func=train)
    for name, func in (('eval-nc', eval_nc), ('eval-lp', eval_lp)):
        q = subs.add_parser(name, help='Evaluate a trained checkpoint')
        context_options(q, sources=False); runtime_options(q); q.add_argument('--model', type=Path, default=DEFAULT_MODEL)
        q.add_argument('--target', default='Toys'); q.add_argument('--checkpoint', type=Path, required=True)
        q.add_argument('--output-dir', type=Path, required=True); q.add_argument('--max-samples', type=int, default=-1)
        if name == 'eval-lp':
            q.add_argument('--sources', nargs=2, default=['Grocery', 'Movies'])
            q.add_argument('--eval-mode', choices=['logit', 'generate'], default='logit')
        q.set_defaults(func=func)
    return p


def main():
    args = parser().parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
