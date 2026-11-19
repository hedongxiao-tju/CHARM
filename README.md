# CHARM

Official implementation of [CHARM: A Multimodal Graph Foundation Model with Hierarchical Context Modeling for Zero-Shot Transfer](https://arxiv.org/abs/2607.26023), accepted by KDD 2027.

CHARM supports node classification and link prediction on multimodal graphs. Model code is in `model/`, and `charm_pipeline` provides commands for data preparation, training, and evaluation.

## Installation

```bash
python -m pip install -e .
python -m charm_pipeline --help
```

Prepare a compatible base model, vision encoder, and MAGB data, then configure their paths as described in the [usage guide](charm_pipeline/README.md).

## Weights

Node classification checkpoints and training initialization weights: [Google Drive](https://drive.google.com/drive/folders/1Tno6sclW7tlsopFEDbvCH210mu1dgK23). Extract the downloaded files into `charm_pipeline/weights/`, preserving the directories inside the archives.

See the [weights guide](charm_pipeline/weights/README.md) for checkpoints and evaluation logs.
