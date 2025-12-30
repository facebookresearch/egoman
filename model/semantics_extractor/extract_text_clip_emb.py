# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
import re
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ============================
# 1. Load annotations and build phrase list
# ============================
def load_phrases_from_pkl(pkl_path: str) -> List[Dict]:
    """
    Load original annotation PKL and extract unique (key, phrase) pairs.

    Each anno item example:
    {
        "image": "frame_000123.jpg",
        "value": [..., "right hand place dumpling bag", ...],
        ...
    }
    """
    all_anno = pickle.load(open(pkl_path, "rb"))
    phrases = []
    seen = set()

    for anno in all_anno:
        phrase_str = anno["value"][2]
        key = anno["image"] + "_" + phrase_str
        if key not in seen:
            seen.add(key)
            phrases.append({"key": key, "phrase": phrase_str})
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
    phrase = phrase[0].lower() + phrase[1:] if phrase else phrase

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
    batch_size: int = 1024,
):
    dataset = PhraseDataset(phrase_list)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model (multi-GPU)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    model = torch.nn.DataParallel(model)
    processor = CLIPProcessor.from_pretrained(model_name)

    emb_dict = {}

    with torch.no_grad():
        for keys, phrases in tqdm(dataloader, total=len(dataloader)):
            # Clean text
            captions = [normalize_phrase(p) for p in phrases]

            # Tokenize & encode
            inputs = processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_features = model.module.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Save each embedding
            for k, caption, emb in zip(keys, captions, text_features):
                emb_dict[k] = {
                    "text": caption,
                    "emb": emb.cpu().numpy().reshape(-1).tolist(),
                }
    # Save final dictionary
    with open(output_pkl, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"Saved {len(emb_dict)} embeddings → {output_pkl}")


# ============================
# 5. Main
# ============================
if __name__ == "__main__":

    # load phrase list from pkl
    # anno_pkl = "../data/egoman_final/egoman_test_final.pkl"
    # phrase_list = load_phrases_from_pkl(anno_pkl)
    # print(f"Loaded {len(phrase_list)} unique act.phrase entries.")

    # create phrase_list from examples
    data_root = "data/examples"
    phrase_list = []
    for image_name in os.listdir(data_root):
        if image_name.endswith(".jpg"):
            phrase_str = image_name[:-4].replace("_", " ")
            print(phrase_str)
            key = image_name + "_" + phrase_str
            phrase_list.append({"key": key, "phrase": phrase_str})

    output_pickle = "../data/examples_text_features.pkl"

    encode_and_save_embeddings(
        phrase_list=phrase_list,
        output_pkl=output_pickle,
        model_name="openai/clip-vit-large-patch14",
        batch_size=512,
    )
