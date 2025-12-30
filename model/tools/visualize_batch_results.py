# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Visualize predicted hand trajectories from inference results.
Loads pkl from inference_new_stage2_od_6dof_fast_robohack.py,
visualizes K predicted hand trajectories, and saves to K images.
"""
import json
import os

import _init_paths

from scipy.spatial.transform import Rotation as R

from utils.visualization_utils import visualize_predictions

if __name__ == "__main__":
    video_id = "examples"
    model_name = "EgoMAN-7B"
    img_dir = f"../data/{video_id}"

    result_pkl_path = f"../output/{model_name}-{video_id}.pkl"
    output_dir = f"../output/visualizations/{video_id}"

    os.makedirs(output_dir, exist_ok=True)
    # Optional: Camera parameters from egoman dataset

    # Load camera parameters if provided
    cam_params_dict = {}
    for file_name in os.listdir(img_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(img_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                cam_params_dict[file_name.split("+")[0]] = data
    if len(cam_params_dict) == 0:
        print("No camera parameters found")
        cam_params_dict = None

    # Run visualization
    visualize_predictions(
        img_dir,
        result_pkl_path=result_pkl_path,
        output_dir=output_dir,
        cam_params_dict=cam_params_dict,
        K=3,  # plot K predicted hand trajectories
    )
