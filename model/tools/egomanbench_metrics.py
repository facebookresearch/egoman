# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import _init_paths
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from utils.eval_utils.metrics import calculate_traj_metrics
from utils.eval_utils.metrics_summary import build_table, scale_distance_metrics
from utils.rot_utils import align_quat_to_styleA, quat_to_rot6d


res_path = "../output/EgoMAN-7B-egomanbench.pkl"
file = res_path.split("/")[-1].split(".")[0]
all_res_list = pickle.load(open(res_path, "rb"))
all_pred_results = []
sample_result = {}
final_target_dir = "../output/"
os.makedirs(final_target_dir, exist_ok=True)
final_res_path = f"{final_target_dir}/{file}_processed.pkl"

if "all_sample_results" in all_res_list:
    all_res_list = all_res_list["all_sample_results"]

if not os.path.exists(final_res_path):
    for res in tqdm(all_res_list):
        sample = res
        if "value" not in sample:
            sample = annotations[sample["image"]]
        if sample["dataset"] not in ["egoexo", "nymeria"]:
            all_traj = np.hstack(
                [
                    np.array(sample["left_hand_traj"])[..., :3],
                    np.array(sample["right_hand_traj"])[..., :3],
                ]
            )[:25]
            all_quat = np.hstack(
                [
                    align_quat_to_styleA(
                        np.array(sample["left_hand_traj"])[..., 3:], "left"
                    ),
                    align_quat_to_styleA(
                        np.array(sample["right_hand_traj"])[..., 3:],
                        "right",
                    ),
                ]
            )[:25]
            sample["value"] = [
                None,
                np.round(all_traj[:5], 3),
                f"{sample['manipulating_hand']} hand manipulates {sample['object_name']}",
                np.round(all_traj, 3),
                np.round(all_quat, 4),
            ]
            sample["benchmark_split"] = "hot3d_ood"
        else:
            sample["benchmark_split"] = "egoman_unseen"

        fut_traj = torch.zeros(50, 6) - 1000
        fut_quat = torch.zeros(50, 12) - 1000
        fut_traj_ori = torch.from_numpy(sample["value"][3])[5:]
        fut_traj[: len(fut_traj_ori)] = fut_traj_ori
        fut_quat_ori = torch.from_numpy(sample["value"][4])[5:]
        fut_quat_ori = fut_quat_ori.reshape(-1, 4)

        # xyzw to wxyz
        fut_quat_ori = torch.stack(
            [
                fut_quat_ori[:, 3],
                fut_quat_ori[:, 0],
                fut_quat_ori[:, 1],
                fut_quat_ori[:, 2],
            ],
            dim=1,
        )
        fut_quat_ori = quat_to_rot6d.forward(fut_quat_ori).reshape(-1, 12)
        fut_quat[: len(fut_quat_ori)] = fut_quat_ori

        # Only compare on valid timesteps (not -1000)
        if "ori_pred_wrist_pose" in res:
            pred_hands = np.zeros((10, 50, 18)) - 1000
            preds = np.array(res["ori_pred_wrist_pose"])
            pred_hands[:, : preds.shape[1]] = preds
        else:
            pred_hands = np.array(
                [single_res["predicted_trajectory"] for single_res in res["k_samples"]]
            )
            pred_hands = pred_hands.transpose(0, 2, 1, 3).reshape(-1, 50, 18)

        metrics_result = calculate_traj_metrics(pred_hands, fut_traj, fut_quat)
        sample_result["trajectory_metrics"] = metrics_result

        sample_result["closest_pred_hands_id"] = metrics_result["best_idx"]
        sample_result["closest_pred_hands"] = pred_hands[
            metrics_result["best_idx"]
        ].tolist()
        sample_result["image"] = sample["image"]
        sample_result["value"] = sample["value"]
        sample_result["benchmark_split"] = sample["benchmark_split"]
        all_pred_results.append(sample_result.copy())

    pickle.dump(all_pred_results, open(final_res_path, "wb"))
    print(final_res_path)

# summarize metrics
k_cap_values = [1, 5, 10]
all_dfs = []

for k_cap in k_cap_values:
    df = build_table(
        final_res_path,
        hand="both",
        k_cap=k_cap,
        benchmark_name="egoman_all",
    )
    df = scale_distance_metrics(df, factor=1.0)  # ADE/FDE/DTW in meters

    if not df.empty:
        # Add K_CAP column
        df.insert(2, "#K", k_cap)
        # Collect for unified table
        all_dfs.append(df)

# Display and save unified table with all K_CAP values
if all_dfs:
    unified_df = pd.concat(all_dfs, ignore_index=True)

    print("\n" + "=" * 120)
    print(f"Unified Metrics Table - All K_CAP Values")
    print("=" * 120)

    # Pretty print unified table
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        lambda x: f"{x:0.3f}",
    ):
        print(unified_df.to_string(index=False))

    # Save to CSV with rounding
    out_path = f"../output/metrics_egoman_all.csv"
    # Round numeric columns to 3 decimal places
    df_to_save = unified_df.copy()
    numeric_cols = ["ADE", "FDE", "DTW", "ROT"]
    for col in numeric_cols:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].round(3)
    df_to_save.to_csv(out_path, index=False)
    print(f"\n{'='*100}")
    print(f"Saved unified table to {out_path}")
    print(f"Total rows: {len(unified_df)}")
