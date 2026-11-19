# Node Classification Weights and Logs

[Google Drive weights folder](https://drive.google.com/drive/folders/1Tno6sclW7tlsopFEDbvCH210mu1dgK23)

Extract the downloaded files into this directory:

```text
weights/
  paper_nc_gm_to_toys/       Grocery + Movies → Toys
  paper_nc_gt_to_movies/     Grocery + Toys → Movies
  paper_nc_tm_to_grocery/    Toys + Movies → Grocery
  initial/mm_projector.bin   Training initialization weights
```

Each checkpoint directory contains training logs in `logs/train/` and test logs in `logs/test/`. Set the base model with `CHARM_MODEL` or `--model`.

## Evaluation example

```bash
python -m charm_pipeline eval-nc \
  --target Toys \
  --checkpoint charm_pipeline/weights/paper_nc_gm_to_toys \
  --output-dir logs/charm_pipeline/paper_toys_nc --gpu 0
```
