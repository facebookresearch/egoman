# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Extract CLIP text embeddings for action phrases from training data.

This script processes action phrases from annotation files and generates
CLIP-L/14 embeddings (768-dimensional vectors) for each unique (image, action) pair.

Output:
    ../data/egoman_dataset/act_emb_dict.pkl - Training set embeddings
    ../data/egoman_dataset/act_emb_val_dict.pkl - Validation set embeddings

    Dictionary format: {
        "image_path_action_phrase": {
            "text": "A photo of action phrase.",
            "emb": [0.123, -0.456, ...]  # list of 768 floats
        }
    }

Usage:
    cd model
    python semantics_extractor/extract_act_emb_clip_train.py
"""

import os
import pickle
import re
from collections import OrderedDict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ============================
# 1. Load annotations and build phrase list
# ============================
def load_phrases_from_pkl(pkl_path: str, split_name: str) -> List[Dict]:
    """
    Load annotation PKL and extract unique (image, action_phrase) pairs.

    Args:
        pkl_path: Path to annotation pickle file
        split_name: Name of the split (e.g., "train", "val")

    Returns:
        List of dicts with keys: "key", "phrase"
        where key = image_path + "_" + action_phrase
    """
    if not os.path.exists(pkl_path):
        print(f"Warning: File not found: {pkl_path}")
        return []

    print(f"Loading annotations from {pkl_path}...")
    all_anno = pickle.load(open(pkl_path, "rb"))
    print(f"  Found {len(all_anno)} samples in {split_name}")

    phrases = []
    seen = set()

    for anno in all_anno:
        if "value" not in anno or len(anno["value"]) < 3:
            continue

        image_path = anno.get("image", "")
        phrase_str = anno["value"][2]  # Action phrase is at index 2

        # Create key: image_path + "_" + action_phrase (no spaces around _)
        key = image_path + "_" + phrase_str

        if key not in seen:
            seen.add(key)
            phrases.append({"key": key, "phrase": phrase_str, "image": image_path})

    print(f"  Extracted {len(phrases)} unique (image, action) pairs")
    return phrases


# ============================
# 2. Text normalization
# ============================
def normalize_phrase(phrase: str) -> str:
    """
    Make the phrase CLIP-friendly:
    - REMOVE any text inside parentheses
    - Clean spacing
    - Add natural prefix for CLIP-style caption
    """
    # Remove parentheses and their content
    phrase = re.sub(r"\([^)]*\)", "", phrase)

    # Clean up spaces
    phrase = re.sub(r"\s+", " ", phrase).strip()

    # Lowercase first letter (optional but keeps style consistent)
    if phrase:
        phrase = phrase[0].lower() + phrase[1:]

    # Add CLIP-style prefix
    return f"A photo of {phrase}."


# ============================
# 3. Dataset for batching
# ============================
class PhraseDataset(Dataset):
    def __init__(self, phrases: List[Dict]):
        self.phrases = phrases

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        item = self.phrases[idx]
        return item["key"], item["phrase"]


# ============================
# 4. Encode and save embeddings
# ============================
def encode_and_save_embeddings(
    phrase_list: List[Dict],
    output_pkl: str,
    model_name: str = "openai/clip-vit-large-patch14",
    batch_size: int = 512,
):
    """
    Extract CLIP embeddings for all phrases and save to pickle file.

    Args:
        phrase_list: List of phrase dictionaries
        output_pkl: Output pickle file path
        model_name: HuggingFace CLIP model name
        batch_size: Batch size for processing
    """
    if len(phrase_list) == 0:
        print(f"No phrases to process for {output_pkl}")
        return

    dataset = PhraseDataset(phrase_list)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        model_module = model.module
    else:
        model_module = model

    processor = CLIPProcessor.from_pretrained(model_name)

    emb_dict = OrderedDict()

    print(f"Extracting CLIP embeddings...")
    with torch.no_grad():
        for keys, phrases in tqdm(dataloader, total=len(dataloader)):
            # Normalize text
            captions = [normalize_phrase(p) for p in phrases]

            # Tokenize & encode
            inputs = processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            text_features = model_module.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Save each embedding
            for k, caption, emb in zip(keys, captions, text_features):
                emb_dict[k] = {
                    "text": caption,
                    "emb": emb.cpu().numpy().reshape(-1).tolist(),
                }

    # Save final dictionary
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"✓ Saved {len(emb_dict)} embeddings → {output_pkl}")


# ============================
# 5. Main
# ============================
if __name__ == "__main__":
    data_root = "../data/egoman_dataset"

    print("=" * 80)
    print("Extracting CLIP Action Embeddings for EgoMAN Training")
    print("=" * 80)

    # Configuration
    annotation_files = {
        "train": ["egoman_pretrain.pkl", "egoman_finetune.pkl"],
        "val": ["egoman-test-final.pkl"],
    }

    output_files = {
        "train": os.path.join(data_root, "act_emb_dict.pkl"),
        "val": os.path.join(data_root, "act_emb_val_dict.pkl"),
    }

    # Process train and val splits separately
    for split_name, anno_files in annotation_files.items():
        print("\n" + "=" * 80)
        print(f"Processing {split_name.upper()} split")
        print("=" * 80)

        # Collect phrases from all annotation files for this split
        all_phrases = []
        for anno_file in anno_files:
            anno_path = os.path.join(data_root, anno_file)
            phrases = load_phrases_from_pkl(anno_path, split_name)
            all_phrases.extend(phrases)

        # Remove duplicates across files
        seen_keys = set()
        unique_phrases = []
        for phrase_dict in all_phrases:
            if phrase_dict["key"] not in seen_keys:
                seen_keys.add(phrase_dict["key"])
                unique_phrases.append(phrase_dict)

        print(f"\nTotal unique phrases for {split_name}: {len(unique_phrases)}")

        if len(unique_phrases) > 0:
            # Extract and save embeddings
            encode_and_save_embeddings(
                phrase_list=unique_phrases,
                output_pkl=output_files[split_name],
                model_name="openai/clip-vit-large-patch14",
                batch_size=512,
            )
        else:
            print(f"Warning: No phrases found for {split_name} split")

    print("\n" + "=" * 80)
    print("CLIP embedding extraction complete!")
    print("=" * 80)
    print("\nGenerated files:")
    for split_name, output_path in output_files.items():
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                data = pickle.load(f)
            print(f"  ✓ {split_name}: {output_path} ({len(data)} embeddings)")
        else:
            print(f"  ✗ {split_name}: {output_path} (not created)")

    print("\nEmbeddings are ready for training!")
