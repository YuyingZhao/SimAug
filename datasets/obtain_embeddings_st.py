"""Compute sentence-transformer embeddings for item titles."""

import argparse
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def parse_args():
    """Parse CLI arguments for embedding generation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--flag", type=str, default="6")
    return parser.parse_args()

def main():
    """Run embedding generation for the selected dataset and model size.

    Returns:
        None
    """
    args = parse_args()
    if args.flag == "6":
        model_path = "all-MiniLM-L6-v2"
    elif args.flag == "12":
        model_path = "all-MiniLM-L12-v2"
    elif args.flag == "mpnet":
        model_path = "all-mpnet-base-v2"
    save_filename = './{}/preprocessed/embs_sentence_transformer_{}.pth'.format(args.dataset_name, args.flag)

    print(args.dataset_name)
    print("LM model path:", model_path)
    print("Save at:", save_filename)
        
    dataset_path = './{}/preprocessed/idx_to_title.pkl'.format(args.dataset_name)

    all_items = []
    with open(dataset_path, 'rb') as file:
        idx_to_title = pickle.load(file)
        for idx in range(len(idx_to_title)): # fix the order
            all_items.append(idx_to_title[idx])
    print('Number of items:', len(all_items))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_path)
    model = model.to(device)

    embeddings = model.encode(all_items)
    print(embeddings.shape)
    print("Statistics:", np.mean(embeddings), np.min(embeddings), np.max(embeddings))
    torch.save(embeddings, save_filename)
    print('Finish saving the embeddings')

if __name__ == "__main__":
    main()
