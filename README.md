# SimAug

This repo introduces **SimAug**, a lightweight, plug-and-play pre-processing method that uses pretrained language models (PLMs) to leverage item text and augment interaction data via textual similarity.

## Overview

This repo contains:
- Dataset preprocessing, text embedding, and augmentation tools under `datasets/`.
- Recommendation models, training, and evaluation code under `rec/`.

## Requirements
- `PyTorch`
- `sentence-transformers`
- `torch_geometric`
- `torch-scatter`

## Installation

Create a virtual environment and install dependencies:

```bash
pip install torch
pip install -U sentence-transformers
pip install torch_geometric
pip install torch-scatter
```

## Dataset Preparation

1) Download Amazon Reviews 2023 data  
Download metadata from the dataset homepage and interaction data from the 0-core page:

- Metadata: https://amazon-reviews-2023.github.io/
- Interactions: https://amazon-reviews-2023.github.io/data_processing/0core.html

Save the two downloaded files into:
```
./datasets/{dataset_name}/src
```
Example filenames: `meta_Baby_Products.jsonl.gz`, `Baby_Products.csv.gz`

2) Preprocess
```bash
cd datasets
python preprocess_kcore.py
```
Preprocessed files are written to `./datasets/{dataset_name}/preprocessed`, including:

- `edges.pkl`: `(train_edges, val_edges, test_edges)`
- `id_to_idx.pkl`: `(user_id_to_idx, item_id_to_idx)`
- `idx_to_title.pkl`: `{item_idx: title}`
- `item_popularity_label.pkl`: popular/unpopular item sets
- `user_activeness_label.pkl`: active/inactive user sets

3) Obtain item text embeddings
```bash
python obtain_embeddings_st.py
```
Embeddings are saved into the preprocessed directory.

4) Generate augmented interactions
```bash
python augment_dataset.py
```
Augmented interactions are saved to `./datasets/{dataset_name}/aug/`.

See `run.sh` for end-to-end examples.

## Training

Train recommendation models from `rec/`:

```bash
cd rec
python main.py
```

## Evaluation

Run evaluation from `rec/`:

```bash
python test.py
```

Results are saved automatically to the output folders described below.

## Project Structure

```
datasets/
  preprocess_kcore.py          # preprocess raw Amazon data
  obtain_embeddings_st.py       # item text embeddings via sentence-transformers
  augment_dataset.py            # generate augmented interactions
  src/                          # raw downloads
  preprocessed/                 # structured data
  aug/                          # augmented interactions

rec/
  parse.py                      # CLI parameters
  main.py                       # training entry point
  dataprocess.py                # data processing for training
  model.py                      # models (LightGCN, MF, ...)
  evaluation.py                 # evaluation metrics
  test.py                       # evaluation runner
  utils.py                      # utilities
  trained_models/               # saved models
  results_rec_list/             # recommendation lists (save_rec.py)
  results_logs/                 # utility & fairness logs (one line per seed)
  results/                      # intermediate files
```

## Notes

- The command examples for all stages are in `run.sh`.
- The augmentation strategies include `random` and `sentence_transformer_6`.
