# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

import torch
from scipy.spatial.transform import Rotation as R


def final_displacement_error(gen_traj: np.array, gt_traj: np.array) -> float:
    """Calculate Final Displacement Error (FDE) between generated and ground truth trajectories"""
    len_gen = gen_traj.shape[0]
    len_gt = gt_traj.shape[0]

    if len_gen > len_gt:
        gen_traj_padded = gen_traj[:len_gt, :]
    elif len_gen < len_gt:
        pad_size = len_gt - len_gen
        last_point = gen_traj[-1, :].reshape(1, -1)
        padding = np.repeat(last_point, pad_size, axis=0)
        gen_traj_padded = np.vstack([gen_traj, padding])
    else:
        gen_traj_padded = gen_traj

    final_gen = gen_traj_padded[-1]
    final_gt = gt_traj[-1]

    diff = np.linalg.norm(final_gt - final_gen, ord=2)

    return diff


def average_displacement_error(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """Calculate Average Displacement Error (ADE) between generated and ground truth trajectories"""
    len_gen = gen_traj.shape[0]
    len_gt = gt_traj.shape[0]

    if len_gen > len_gt:
        gen_traj_padded = gen_traj[:len_gt, :]
    elif len_gen < len_gt:
        pad_size = len_gt - len_gen
        last_point = gen_traj[-1, :].reshape(1, -1)
        padding = np.repeat(last_point, pad_size, axis=0)
        gen_traj_padded = np.vstack([gen_traj, padding])
    else:
        gen_traj_padded = gen_traj

    diff = np.linalg.norm(gt_traj - gen_traj_padded, ord=2, axis=1).mean()

    return diff


def dtw_trajectory_distance(
    gen_traj: np.ndarray,
    gt_traj: np.ndarray,
    *,
    band: int | None = None,  # Sakoe–Chiba window; e.g., 10 or None for full
    normalize: bool = True,
    return_path: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Dynamic Time Warping (DTW) distance between two trajectories (N,D) and (M,D).
    Local cost = Euclidean distance. Normalizes by *warping path length* if requested.

    Args:
        gen_traj: (N, D) array
        gt_traj:  (M, D) array
        band:     optional Sakoe–Chiba radius so |i-j| <= band
        normalize: if True, divide total cost by warping path length
        return_path: if True, also return the optimal (P,2) path indices

    Returns:
        dtw: float (average per-step DTW cost if normalize=True, else total cost)
        path (optional): (P,2) int array of aligned indices (i,j), 0-based
    """
    gen_traj = np.asarray(gen_traj, dtype=float)
    gt_traj = np.asarray(gt_traj, dtype=float)

    if gen_traj.ndim != 2 or gt_traj.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (N,D) and (M,D).")
    if gen_traj.shape[1] != gt_traj.shape[1]:
        raise ValueError(
            f"Dim mismatch: D1={gen_traj.shape[1]} vs D2={gt_traj.shape[1]}"
        )

    n, m = len(gen_traj), len(gt_traj)
    if n == 0 or m == 0:
        return (np.inf, np.empty((0, 2), int)) if return_path else np.inf

    # DP matrix initialized with +inf; origin 0
    dtw = np.full((n + 1, m + 1), np.inf, dtype=float)
    dtw[0, 0] = 0.0

    # Optional band -> restrict j for each i
    for i in range(1, n + 1):
        if band is None:
            j_start, j_end = 1, m
        else:
            j_center = int(np.round(i * m / n))  # rough diagonal; optional
            j_start = max(1, i - band)
            j_end = min(m, i + band)

        # Iterate j within window
        for j in range(j_start, j_end + 1):
            cost = np.linalg.norm(gen_traj[i - 1] - gt_traj[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]  # insertion  # deletion
            )  # match

    total = dtw[n, m]

    # Backtrack to get the actual warping path length
    path_len = None
    path = None
    if normalize or return_path:
        i, j = n, m
        steps = []
        while i > 0 or j > 0:
            steps.append((i - 1, j - 1))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # choose predecessor with min cost
                a, b, c = dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]
                move = 2  # diag by default
                if a <= b and a <= c:
                    move = 0  # up
                elif b <= a and b <= c:
                    move = 1  # left
                if move == 0:  # up
                    i -= 1
                elif move == 1:  # left
                    j -= 1
                else:  # diag
                    i -= 1
                    j -= 1
        steps.reverse()
        path = np.asarray(steps, dtype=int)
        path_len = max(len(path), 1)

    if normalize:
        total = total / path_len

    if return_path:
        return float(total), path
    return float(total)


def rotation_6d_to_matrix_numpy(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to rotation matrix using NumPy.

    Args:
        rot_6d: [..., 6] 6D rotation representation

    Returns:
        [..., 3, 3] rotation matrix
    """
    # Reshape to get two 3D vectors
    x_raw = rot_6d[..., :3]  # First column
    y_raw = rot_6d[..., 3:6]  # Second column (unnormalized)

    # Normalize first column
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)

    # Gram-Schmidt orthogonalization for second column
    y = y_raw - np.sum(x * y_raw, axis=-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)

    # Third column via cross product
    z = np.cross(x, y, axis=-1)

    # Stack into matrix
    matrix = np.stack([x, y, z], axis=-1)  # [..., 3, 3]

    return matrix


def calculate_traj_metrics(
    pred_wrist_pose: np.ndarray,
    gt_wrist_pose: np.ndarray,
    gt_rotation: np.ndarray = None,
):
    """
    Calculate trajectory metrics for predicted vs ground truth wrist poses and rotations

    Args:
        pred_wrist_pose: [K, L, 18] predicted hand trajectories (9D per hand: 3 pos + 6 rot)
        gt_wrist_pose: [L, 6] ground truth hand position trajectories
        gt_rotation: [L, 12] ground truth hand rotations (6D per hand), optional

    Returns:
        Dictionary containing metrics for each K sample and summary statistics
    """
    if isinstance(pred_wrist_pose, torch.Tensor):
        pred_wrist_pose = pred_wrist_pose.cpu().numpy()
    if isinstance(gt_wrist_pose, torch.Tensor):
        gt_wrist_pose = gt_wrist_pose.cpu().numpy()
    if gt_rotation is not None and isinstance(gt_rotation, torch.Tensor):
        gt_rotation = gt_rotation.cpu().numpy()

    K = pred_wrist_pose.shape[0]  # Number of samples
    L = pred_wrist_pose.shape[1]  # Sequence length

    # Create valid mask (exclude -1000 values)
    valid_mask = (gt_wrist_pose != -1000).any(axis=-1)  # [L]

    # Apply valid mask to ground truth
    gt_valid = gt_wrist_pose[valid_mask]  # [L_valid, 6]

    if len(gt_valid) == 0:
        empty_metrics = {
            "avg_distance": 0.0,
            "left_hand_ade": 0.0,
            "left_hand_fde": 0.0,
            "left_hand_dtw": 0.0,
            "right_hand_ade": 0.0,
            "right_hand_fde": 0.0,
            "right_hand_dtw": 0.0,
            "left_rot_error": 0.0,
            "right_rot_error": 0.0,
            "overall_rot_error": 0.0,
        }
        return {
            "k_samples": [],
            "mean_metrics": empty_metrics,
            "std_metrics": empty_metrics,
            "best_metrics": {**empty_metrics, "k": 0},
        }

    # Split ground truth positions into left and right hands
    gt_left_pos = gt_valid[:, :3]  # [L_valid, 3]
    gt_right_pos = gt_valid[:, 3:]  # [L_valid, 3]

    # Split ground truth rotations if provided
    gt_left_rot = None
    gt_right_rot = None
    if gt_rotation is not None:
        gt_rotation_valid = gt_rotation[valid_mask]  # [L_valid, 12]
        gt_left_rot = gt_rotation_valid[:, :6]  # [L_valid, 6]
        gt_right_rot = gt_rotation_valid[:, 6:]  # [L_valid, 6]

    # Storage for K sample results
    k_sample_results = []

    # Calculate metrics for each K sample
    for k in range(K):
        pred_k = pred_wrist_pose[k]  # [L, 18]
        pred_k_valid = pred_k[valid_mask]  # [L_valid, 18]

        if len(pred_k_valid) == 0:
            continue

        # Reshape prediction: [L_valid, 18] -> [L_valid, 2, 9] -> [2, L_valid, 9]
        pred_k_reshaped = pred_k_valid.reshape(-1, 2, 9).transpose(1, 0, 2)

        # Split prediction into position and rotation components
        pred_left_pos = pred_k_reshaped[0, :, :3]  # [L_valid, 3]
        pred_left_rot = pred_k_reshaped[0, :, 3:9]  # [L_valid, 6]
        pred_right_pos = pred_k_reshaped[1, :, :3]  # [L_valid, 3]
        pred_right_rot = pred_k_reshaped[1, :, 3:9]  # [L_valid, 6]

        # Calculate position metrics for left hand
        left_ade = 0.0
        left_fde = 0.0
        left_dtw = 0.0
        if len(pred_left_pos) > 0 and len(gt_left_pos) > 0:
            left_ade = average_displacement_error(pred_left_pos, gt_left_pos)
            left_fde = final_displacement_error(pred_left_pos, gt_left_pos)
            left_dtw = dtw_trajectory_distance(pred_left_pos, gt_left_pos)

        # Calculate position metrics for right hand
        right_ade = 0.0
        right_fde = 0.0
        right_dtw = 0.0
        if len(pred_right_pos) > 0 and len(gt_right_pos) > 0:
            right_ade = average_displacement_error(pred_right_pos, gt_right_pos)
            right_fde = final_displacement_error(pred_right_pos, gt_right_pos)
            right_dtw = dtw_trajectory_distance(pred_right_pos, gt_right_pos)

        # Calculate rotation errors
        left_rot_error = 0.0
        right_rot_error = 0.0

        if gt_left_rot is not None and len(pred_left_rot) > 0:
            rot_errors = []
            for t in range(len(pred_left_rot)):
                try:
                    pred_rot_matrix = rotation_6d_to_matrix_numpy(
                        pred_left_rot[t].astype(np.float32)
                    )
                    gt_rot_matrix = rotation_6d_to_matrix_numpy(
                        gt_left_rot[t].astype(np.float32)
                    )
                    # Compute relative rotation matrix
                    relative_rot = np.matmul(pred_rot_matrix, gt_rot_matrix.T)
                    # Compute rotation angle error using trace
                    trace = np.trace(relative_rot)
                    trace = np.clip(trace, -1.0, 3.0)
                    angle_error = np.arccos((trace - 1) / 2.0)
                    rot_errors.append(angle_error)
                except:
                    continue
            if rot_errors:
                left_rot_error = np.mean(rot_errors) * 180.0 / np.pi

        if gt_right_rot is not None and len(pred_right_rot) > 0:
            rot_errors = []
            for t in range(len(pred_right_rot)):
                try:
                    pred_rot_matrix = rotation_6d_to_matrix_numpy(
                        pred_right_rot[t].astype(np.float32)
                    )
                    gt_rot_matrix = rotation_6d_to_matrix_numpy(
                        gt_right_rot[t].astype(np.float32)
                    )
                    # Compute relative rotation matrix
                    relative_rot = np.matmul(pred_rot_matrix, gt_rot_matrix.T)
                    # Compute rotation angle error using trace
                    trace = np.trace(relative_rot)
                    trace = np.clip(trace, -1.0, 3.0)
                    angle_error = np.arccos((trace - 1) / 2.0)
                    rot_errors.append(angle_error)
                except:
                    continue
            if rot_errors:
                right_rot_error = np.mean(rot_errors) * 180.0 / np.pi

        # Calculate overall rotation error
        overall_rot_error = 0.0
        rot_valid_count = 0
        if left_rot_error > 0:
            overall_rot_error += left_rot_error
            rot_valid_count += 1
        if right_rot_error > 0:
            overall_rot_error += right_rot_error
            rot_valid_count += 1
        if rot_valid_count > 0:
            overall_rot_error /= rot_valid_count

        # Overall average distance (ADE across both hands)
        total_valid_points = len(pred_left_pos) + len(pred_right_pos)
        avg_distance = 0.0
        if total_valid_points > 0:
            if len(pred_left_pos) > 0 and len(pred_right_pos) > 0:
                avg_distance = (
                    left_ade * len(pred_left_pos) + right_ade * len(pred_right_pos)
                ) / total_valid_points
            elif len(pred_left_pos) > 0:
                avg_distance = left_ade
            elif len(pred_right_pos) > 0:
                avg_distance = right_ade

        # Store results for this k-th sample
        k_result = {
            "k": k,
            "avg_distance": avg_distance,
            "left_hand_ade": left_ade,
            "left_hand_fde": left_fde,
            "left_hand_dtw": left_dtw,
            "right_hand_ade": right_ade,
            "right_hand_fde": right_fde,
            "right_hand_dtw": right_dtw,
            "left_rot_error": left_rot_error,
            "right_rot_error": right_rot_error,
            "overall_rot_error": overall_rot_error,
        }
        k_sample_results.append(k_result)

    if not k_sample_results:
        empty_metrics = {
            "avg_distance": 0.0,
            "left_hand_ade": 0.0,
            "left_hand_fde": 0.0,
            "left_hand_dtw": 0.0,
            "right_hand_ade": 0.0,
            "right_hand_fde": 0.0,
            "right_hand_dtw": 0.0,
            "left_rot_error": 0.0,
            "right_rot_error": 0.0,
            "overall_rot_error": 0.0,
        }
        return {
            "k_samples": [],
            "mean_metrics": empty_metrics,
            "std_metrics": empty_metrics,
            "best_metrics": {**empty_metrics, "k": 0},
        }

    # Collect metrics across K samples
    k_avg_distances = [r["avg_distance"] for r in k_sample_results]
    k_left_ades = [
        r["left_hand_ade"] for r in k_sample_results if r["left_hand_ade"] > 0
    ]
    k_left_fdes = [
        r["left_hand_fde"] for r in k_sample_results if r["left_hand_fde"] > 0
    ]
    k_left_dtws = [
        r["left_hand_dtw"] for r in k_sample_results if r["left_hand_dtw"] > 0
    ]
    k_right_ades = [
        r["right_hand_ade"] for r in k_sample_results if r["right_hand_ade"] > 0
    ]
    k_right_fdes = [
        r["right_hand_fde"] for r in k_sample_results if r["right_hand_fde"] > 0
    ]
    k_right_dtws = [
        r["right_hand_dtw"] for r in k_sample_results if r["right_hand_dtw"] > 0
    ]
    k_left_rot_errors = [
        r["left_rot_error"] for r in k_sample_results if r["left_rot_error"] > 0
    ]
    k_right_rot_errors = [
        r["right_rot_error"] for r in k_sample_results if r["right_rot_error"] > 0
    ]
    k_overall_rot_errors = [
        r["overall_rot_error"] for r in k_sample_results if r["overall_rot_error"] > 0
    ]

    # Mean metrics
    mean_metrics = {
        "avg_distance": np.mean(k_avg_distances) if k_avg_distances else 0.0,
        "left_hand_ade": np.mean(k_left_ades) if k_left_ades else 0.0,
        "left_hand_fde": np.mean(k_left_fdes) if k_left_fdes else 0.0,
        "left_hand_dtw": np.mean(k_left_dtws) if k_left_dtws else 0.0,
        "right_hand_ade": np.mean(k_right_ades) if k_right_ades else 0.0,
        "right_hand_fde": np.mean(k_right_fdes) if k_right_fdes else 0.0,
        "right_hand_dtw": np.mean(k_right_dtws) if k_right_dtws else 0.0,
        "left_rot_error": np.mean(k_left_rot_errors) if k_left_rot_errors else 0.0,
        "right_rot_error": np.mean(k_right_rot_errors) if k_right_rot_errors else 0.0,
        "overall_rot_error": (
            np.mean(k_overall_rot_errors) if k_overall_rot_errors else 0.0
        ),
    }

    # Std metrics
    std_metrics = {
        "avg_distance": (np.std(k_avg_distances) if len(k_avg_distances) > 1 else 0.0),
        "left_hand_ade": np.std(k_left_ades) if len(k_left_ades) > 1 else 0.0,
        "left_hand_fde": np.std(k_left_fdes) if len(k_left_fdes) > 1 else 0.0,
        "left_hand_dtw": np.std(k_left_dtws) if len(k_left_dtws) > 1 else 0.0,
        "right_hand_ade": np.std(k_right_ades) if len(k_right_ades) > 1 else 0.0,
        "right_hand_fde": np.std(k_right_fdes) if len(k_right_fdes) > 1 else 0.0,
        "right_hand_dtw": np.std(k_right_dtws) if len(k_right_dtws) > 1 else 0.0,
        "left_rot_error": (
            np.std(k_left_rot_errors) if len(k_left_rot_errors) > 1 else 0.0
        ),
        "right_rot_error": (
            np.std(k_right_rot_errors) if len(k_right_rot_errors) > 1 else 0.0
        ),
        "overall_rot_error": (
            np.std(k_overall_rot_errors) if len(k_overall_rot_errors) > 1 else 0.0
        ),
    }

    # Best metrics (metrics from the closest trajectory - lowest average distance)
    if len(k_avg_distances) > 0:
        best_idx = np.argmin(k_avg_distances)
        best_metrics = k_sample_results[best_idx].copy()
    else:
        best_metrics = {
            "k": 0,
            "avg_distance": 0.0,
            "left_hand_ade": 0.0,
            "left_hand_fde": 0.0,
            "left_hand_dtw": 0.0,
            "right_hand_ade": 0.0,
            "right_hand_fde": 0.0,
            "right_hand_dtw": 0.0,
            "left_rot_error": 0.0,
            "right_rot_error": 0.0,
            "overall_rot_error": 0.0,
        }

    return {
        "k_samples": k_sample_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "best_metrics": best_metrics,
        "best_idx": best_idx,
    }


def aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        summary[k + "_mean"] = float(np.mean(vals)) if vals else 0.0
        summary[k + "_std"] = float(np.std(vals)) if vals else 0.0
    return summary
