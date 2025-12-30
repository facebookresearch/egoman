# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle

import _init_paths
from semantics_extractor.extract_vis_dinov3 import process_batch_images
from tqdm import tqdm


batch_size = 256 * 4
img_data_root = "../data/egomanbench_imgs"
img_paths = [
    os.path.join(img_data_root, f)
    for f in os.listdir(img_data_root)
    if f.endswith(".jpg")
]
features_dict = {}
out_path = f"../data/egomanbench_vis_features.pkl"
if not os.path.exists(out_path):
    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_paths = img_paths[i : i + batch_size]
        feats = process_batch_images(batch_paths)

        # store features for each image
        for p, f in zip(batch_paths, feats):
            features_dict[p.split("/")[-1]] = f
with open(out_path, "wb") as f:
    pickle.dump(features_dict, f)

print(f"Saved features for {len(features_dict)} images to {out_path}")
