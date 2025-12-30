# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle

import _init_paths

import cv2

import torch
from torchvision import transforms
from tqdm import tqdm

# ---------------------------
# 1. Load DINOv3 model
# ---------------------------
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
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)  # tensor [3, 224, 224]


def process_single_image(img_path):
    img = load_and_preprocess(img_path)
    with torch.no_grad():
        feats = dinov3_model(
            img.unsqueeze(0).to("cuda:0")
        )  # [1, D] or [1, tokens, D] depending on model
    return feats.detach().cpu().numpy()


def process_batch_images(img_paths):
    batch_imgs = [load_and_preprocess(p) for p in img_paths]
    batch_tensor = torch.stack(batch_imgs).to("cuda:0")  # [B, 3, 224, 224]
    with torch.no_grad():
        feats = dinov3_model(
            batch_tensor.to("cuda:0")
        )  # [B, D] or [B, tokens, D] depending on model
    return feats.detach().cpu().numpy()


if __name__ == "__main__":
    batch_size = 256 * 4  # adjust based on GPU memory
    data_root = "../data/examples"
    img_paths = [
        os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".jpg")
    ]

    features_dict = {}

    # process in batches
    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_paths = img_paths[i : i + batch_size]
        feats = process_batch_images(batch_paths)

        # store features for each image
        for p, f in zip(batch_paths, feats):
            features_dict[p.split("/")[-1]] = f

    out_path = "../data/examples_vis_features.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Saved features for {len(features_dict)} images to {out_path}")
