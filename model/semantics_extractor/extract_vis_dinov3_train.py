# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Extract DINOv3 visual features from training images for EgoMAN training.

This script processes all images referenced in the training annotation files
and extracts DINOv3-L/16 features (1024-dimensional vectors) for each image.

Output:
    ../data/egoman_dataset/egoman_dinov3_features.pkl

    Dictionary format: {image_path: feature_array}
    where image_path is the relative path as stored in annotation files.

Usage:
    cd model
    python semantics_extractor/extract_vis_dinov3_train.py
"""

import os
import pickle
from collections import OrderedDict

import _init_paths

import cv2

import torch
from torchvision import transforms
from tqdm import tqdm

# ---------------------------
# 1. Load DINOv3 model
# ---------------------------
print("Loading DINOv3 model...")
dinov3_model = (
    torch.hub.load(
        "semantics_extractor/dinov3",
        "dinov3_vitl16",
        source="local",
        weights="../data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    .to("cuda:0")
    .eval()
)
print("DINOv3 model loaded successfully.")

# ---------------------------
# 2. Preprocessing transform
# ---------------------------
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((448, 448), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


# ---------------------------
# 3. Read and preprocess one image
# ---------------------------
def load_and_preprocess(img_path):
    """Load image and preprocess for DINOv3."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)


def process_batch_images(img_paths):
    """Process a batch of images and extract DINOv3 features."""
    batch_imgs = []
    valid_paths = []

    for p in img_paths:
        try:
            img = load_and_preprocess(p)
            batch_imgs.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")
            continue

    if len(batch_imgs) == 0:
        return [], []

    batch_tensor = torch.stack(batch_imgs).to("cuda:0")
    with torch.no_grad():
        feats = dinov3_model(batch_tensor)

    return feats.detach().cpu().numpy(), valid_paths


def collect_all_image_paths(data_root="../data/egoman_dataset"):
    """
    Collect all unique image paths from training annotation files.

    Returns:
        dict: {relative_image_path: full_image_path}
    """
    annotation_files = [
        "egoman_pretrain.pkl",
        "egoman_finetune.pkl",
        "egoman-test-final.pkl",  # optional, for validation
    ]

    image_path_map = OrderedDict()

    for anno_file in annotation_files:
        anno_path = os.path.join(data_root, anno_file)

        if not os.path.exists(anno_path):
            print(f"Warning: Annotation file not found: {anno_path}")
            continue

        print(f"Loading annotations from {anno_file}...")
        annotations = pickle.load(open(anno_path, "rb"))
        print(f"  Found {len(annotations)} samples")

        # Extract image paths
        for anno in annotations:
            if "image" in anno:
                relative_path = anno["image"]
                data_path = anno.get("data_path", data_root)
                full_path = os.path.join(data_path, relative_path)

                # Store with relative path as key (used in training)
                if relative_path not in image_path_map:
                    if os.path.exists(full_path):
                        image_path_map[relative_path] = full_path
                    else:
                        print(f"Warning: Image not found: {full_path}")

    print(f"\nTotal unique images to process: {len(image_path_map)}")
    return image_path_map


if __name__ == "__main__":
    batch_size = 256  # adjust based on GPU memory
    data_root = "../data/egoman_dataset"

    # Collect all image paths from training annotations
    print("=" * 80)
    print("Step 1: Collecting image paths from annotation files")
    print("=" * 80)
    image_path_map = collect_all_image_paths(data_root)

    if len(image_path_map) == 0:
        print("Error: No images found to process!")
        print("Please ensure annotation files exist at:")
        print(f"  - {data_root}/egoman_pretrain.pkl")
        print(f"  - {data_root}/egoman_finetune.pkl")
        print(f"  - {data_root}/egoman-test-final.pkl")
        exit(1)

    # Process images in batches
    print("\n" + "=" * 80)
    print("Step 2: Extracting DINOv3 features")
    print("=" * 80)
    features_dict = {}

    relative_paths = list(image_path_map.keys())
    full_paths = list(image_path_map.values())

    for i in tqdm(range(0, len(full_paths), batch_size), desc="Processing batches"):
        batch_full_paths = full_paths[i : i + batch_size]
        batch_relative_paths = relative_paths[i : i + batch_size]

        feats, valid_paths = process_batch_images(batch_full_paths)

        # Store features with relative paths as keys (matching training data)
        for idx, full_p in enumerate(valid_paths):
            relative_p = batch_relative_paths[batch_full_paths.index(full_p)]
            features_dict[relative_p] = feats[idx]

    # Save features
    print("\n" + "=" * 80)
    print("Step 3: Saving features")
    print("=" * 80)
    out_path = os.path.join(data_root, "egoman_dinov3_features.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"\n✓ Successfully saved features for {len(features_dict)} images")
    print(f"  Output: {out_path}")
    print(f"  Feature shape: {features_dict[list(features_dict.keys())[0]].shape}")
    print("\nFeatures are ready for training!")
