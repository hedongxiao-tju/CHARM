# CHARM Usage Guide

The command-line interface provides data preparation, training, node classification, and link prediction.

## Project structure

```text
charm_pipeline/  Command-line interface and configuration
model/           Model, datasets, and node classification evaluation
charm_train.py   Joint node classification and link prediction training
lp_zeroshot.py   Link prediction evaluation
scripts/         Hierarchical graph, summary, and bridge construction
```

## Setup

Use Python 3.10 and a CUDA-enabled PyTorch environment, and install the project dependencies:

```bash
python -m pip install -e .
```

Prepare a base model, vision encoder, preprocessed graph, and MAGB datasets. Configure their paths:

```bash
export CHARM_DATA_ROOT=/path/to/MAGB
export CHARM_SYNTHETIC_ROOT=/path/to/prepared_synthetic_graph
export CHARM_MODEL=/path/to/base_model
export CHARM_CLIP_MODEL=/path/to/vision_encoder
```

The base model directory contains the model configuration, tokenizer, and weights. The local vision encoder path can also be supplied with `--clip-model` to `prepare-graph` or `train`.

The data directory has the following structure; each dataset uses the same layout as Toys:

```text
MAGB/
  Grocery/  Movies/  Toys/
    <dataset>.csv  clip_feat.pt  labels.pt
    train_mask.pt  val_mask.pt  test_mask.pt
    graph_label_config.json  ppr.pt  lp_zeroshot_split.pt
    ...
  realnode_kmeans_product3_g24_topm3_semantic_cliptext_bridge/
    dataset_offsets.json  kmeans_global_synthetic_graph_meta.json
    modality_complement_bridge_edges.pt  original_graph_edges.pt  ...
```

Download checkpoints and initialization weights from [Google Drive](https://drive.google.com/drive/folders/1Tno6sclW7tlsopFEDbvCH210mu1dgK23), and extract their directories into `charm_pipeline/weights/`. See the [weights guide](weights/README.md) for the checkpoint mapping.

The examples use `Grocery + Movies → Toys`. Use `--gpu` to select a device. `--dry-run` prints the command; remove it to run the command.

## Inspect data

```bash
python -m charm_pipeline check --sources Grocery Movies --target Toys
```

## Prepare data

Build the hierarchical graph, node summaries, and modality-complementary bridges from existing features and PPR data:

```bash
python -m charm_pipeline prepare-graph \
  --datasets Grocery Movies Toys \
  --output-root /path/to/charm_prepared_gmt \
  --gpu 0 --dry-run
```

The final graph is written to `04_bridges/` within the output directory. To regenerate link prediction splits:

```bash
python -m charm_pipeline prepare-splits \
  --datasets Grocery Movies Toys \
  --synthetic-root /path/to/04_bridges --dry-run
```

The default training and validation splits use 50% hard negatives; the test split uses random negatives. Pass `--replace-existing` to rebuild existing splits.

## Joint training

```bash
python -m charm_pipeline train \
  --sources Grocery Movies --target Toys \
  --output-dir checkpoints/charm_pipeline/gm_to_toys \
  --gpu 0 --dry-run
```

Training uses node classification and link prediction samples from the source domains. It selects 5,000 link prediction samples per source domain by default. Adjust the number with `--max-lp-samples`, or set it to `0` to use all samples. Use `--init-adapter` to select initialization weights.

## Node classification

```bash
python -m charm_pipeline eval-nc \
  --target Toys \
  --checkpoint charm_pipeline/weights/paper_nc_gm_to_toys \
  --output-dir logs/charm_pipeline/gm_to_toys_nc --gpu 0 --dry-run
```

Evaluation scores candidate options with option-prior calibration and writes results to `predictions.jsonl`. Use `--max-samples` to limit the number of samples.

## Link prediction

```bash
python -m charm_pipeline eval-lp \
  --sources Grocery Movies --target Toys \
  --checkpoint checkpoints/charm_pipeline/gm_to_toys \
  --output-dir logs/charm_pipeline/gm_to_toys_lp --gpu 0 --dry-run
```

The default evaluator scores answer options using a fixed threshold of 0.5. Set `--eval-mode generate` for generation-based evaluation. Evaluation uses graph neighborhood and PPR caches; for edge-split evaluation, construct the graph and caches from training edges.
