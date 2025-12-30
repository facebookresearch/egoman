# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 6: Trajectory Quality Filtering (Rule-Based)

Purpose:
    Filter out low-quality trajectories using rule-based heuristics to ensure only
    physically plausible and visually observable hand movements are retained in the dataset.
    This is the first stage of quality control before optional GPT-based validation.

Dependencies:
    - Runs after: Step 5 (reason_numeric_qa_generator.py) - requires numeric QA data with trajectories
    - Runs before: Final dataset compilation or optional GPT filtering

Input:
    - Trajectory data from Step 5 containing:
        * value[0]: Waypoint data (start, contact, end) with 2D projections
        * value[2]: Action phrase describing the interaction
        * value[-2]: 3D hand positions over time (wrist_pose)
        * value[-1]: Quaternion orientations over time (wrist_quat)
        * interact: Interaction metadata (approach duration, etc.)

Output:
    - Filtered trajectory list with only valid interactions meeting:
        * Visibility: Hands visible in final frame (and contact frame if exists)
        * Non-trivial movement: Movement distance is valid if displacement > 0.3 m OR rotation is valid if angular change > 60°
        * Approach validation: For grasp/pick actions, requires proper approach stage

Quality Criteria:

1. **Visibility Requirements:**
   - Final frame: Acting hand(s) must be visible in 2D projection
   - Contact frame: Acting hand(s) must be visible if approach stage exists
   - Start frame: Acting hand(s) should be visible (tracked but not strictly enforced)

2. **Movement Thresholds:**
   - Distance categories:
     * Short: < 0.15m
     * Mid: 0.15m - 0.3m
     * Long: > 0.3m
   - Rotation categories:
     * Short: < 30°
     * Mid: 30° - 60°
     * Long: > 60°

3. **Static Interaction Filter:**
   - Reject if (short distance AND not long rotation) OR (short rotation AND not long distance)
   - Exception: Grasp/pick actions require long approach OR mid+ movement

4. **Approach Stage Validation:**
   - For grasp/pick actions:
     * Requires approach stage with duration > 0.5s, OR
     * Must have at least mid-range distance movement
   - Approach categorization:
     * No approach: two_stage = False
     * Short approach: < 0.5s
     * Long approach: ≥ 0.5s

Filtering Functions:
    - max_deg_from_start(): Calculate maximum rotation from initial quaternion
    - quat_angle_gap_xyzw_np(): Compute angular distance between quaternions
    - trajectory_quality_filtering_by_rules(): Main filtering function

Usage:
    valid_trajectories = trajectory_quality_filtering_by_rules(valid_data_list)

    Where valid_data_list is the output from Step 5 numeric QA generation.

Output Statistics (tracked internally):
    - vis_count: Trajectories with visible hands at end
    - contact_vis: Trajectories with visible hands at contact
    - start_vis: Trajectories with visible hands at start
    - all_vis: Trajectories visible at both contact and end
    - valid_count: Final count after all filters

Note: This is a conservative filter designed to remove obvious quality issues.
      For stricter semantic validation, use step6_traj_quality_filter_gpt.py afterward.
"""

import pickle

import numpy as np
import torch
from tqdm import tqdm


def _normalize(q, eps=1e-12):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def max_deg_from_start(q_all_xyzw: np.ndarray):
    """
    q_all_xyzw: (T, 4) in xyzw
    Returns: max_deg (float), max_idx (int), all_degs (T,)
    """
    q_all = _normalize(q_all_xyzw.astype(np.float64))
    q0 = q_all[0]  # start quaternion
    # dot with sign fix (abs handles q ~ -q equivalence)
    dots = np.abs(np.sum(q_all * q0, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    angles_rad = 2.0 * np.arccos(dots)
    angles_deg = np.degrees(angles_rad)
    max_idx = int(np.argmax(angles_deg))
    return float(angles_deg[max_idx]), max_idx, angles_deg


def quat_angle_gap_xyzw_np(q_start, q_end, eps=1e-8):
    """Returns (radians, degrees). Quats are xyzw."""
    q1 = np.asarray(q_start, dtype=float)
    q2 = np.asarray(q_end, dtype=float)
    n1 = max(np.linalg.norm(q1), eps)
    n2 = max(np.linalg.norm(q2), eps)
    q1 /= n1
    q2 /= n2
    # minimal rotation distance; abs handles q ~ -q equivalence
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    angle = 2.0 * np.arccos(abs(dot))
    return angle, np.degrees(angle)


def quat_angle_gap_xyzw_torch(q_start, q_end, eps=1e-8):
    """Returns (radians, degrees). Quats are xyzw. Supports tensors or lists."""
    q1 = torch.as_tensor(q_start, dtype=torch.float32)
    q2 = torch.as_tensor(q_end, dtype=torch.float32)
    q1 = q1 / torch.clamp(q1.norm(p=2), min=eps)
    q2 = q2 / torch.clamp(q2.norm(p=2), min=eps)
    dot = torch.clamp(torch.dot(q1, q2), -1.0, 1.0)
    angle = 2.0 * torch.arccos(torch.abs(dot))
    return angle.item(), torch.rad2deg(angle).item()


def trajectory_quality_filtering_by_rules(valid_data_list):
    """
    Filter out low quality trajectories based on rules.
    We need trajectories:
        1. visible by the end
        2. move at a non-trivial distance and angle range
    Input: valid_data_list (list of dict): a list of trajectory data item
    """
    # check if visible
    valid_count = 0
    vis_count = 0
    contact_vis = 0
    start_vis = 0
    all_vis = 0
    valid_list = []
    approach_valid = {"no": [], "short": [], "long": []}
    distance_valid = {"short": [], "mid": [], "long": []}
    rot_valid = {"short": [], "mid": [], "long": []}
    for cur_idx, cur_data in tqdm(enumerate(valid_data_list)):
        # check if visible
        left_visible = (cur_data["value"][0][..., 7:9] != -1000).sum(-1) // 2
        right_visible = (cur_data["value"][0][..., 9:11] != -1000).sum(-1) // 2
        final_visible = False
        if "left hand" in cur_data["value"][2] and left_visible[-1] > 0:
            final_visible = True
        elif "right hand" in cur_data["value"][2] and right_visible[-1] > 0:
            final_visible = True
        else:
            if left_visible[-1] > 0 and right_visible[-1] > 0:
                final_visible = True
        contact_visible = False
        if "left hand" in cur_data["value"][2] and left_visible[1] > 0:
            contact_visible = True
        elif "right hand" in cur_data["value"][2] and right_visible[1] > 0:
            contact_visible = True
        else:
            if left_visible[1] > 0 and right_visible[1] > 0:
                contact_visible = True

        start_visible = False
        if "left hand" in cur_data["value"][2] and left_visible[0] > 0:
            start_visible = True
        elif "right hand" in cur_data["value"][2] and right_visible[0] > 0:
            start_visible = True
        else:
            if left_visible[0] > 0 and right_visible[0] > 0:
                start_visible = True
        if start_visible:
            start_vis += 1
        if final_visible:
            vis_count += 1
        if contact_visible:
            contact_vis += 1
        if final_visible and contact_visible:
            all_vis += 1
        else:
            continue

        # approach time
        if "approach" in cur_data["interact"]:
            if cur_data["interact"]["approach"]["end_time"] * 4.5 > 0.5:
                approach_tag = "long"
            else:
                approach_tag = "short"
        else:
            approach_tag = "no"

        # move distance
        left_quat_all = cur_data["value"][-1][5:, :4]  # xyzw
        right_quat_all = cur_data["value"][-1][5:, 4:]  # xyzw
        left_max_deg, left_idx, left_all_deg = max_deg_from_start(left_quat_all)
        right_max_deg, right_idx, right_all_deg = max_deg_from_start(right_quat_all)

        left_dist = np.linalg.norm(
            cur_data["value"][-2][-1][:3] - cur_data["value"][-2][5][:3]
        )
        right_dist = np.linalg.norm(
            cur_data["value"][-2][-1][3:] - cur_data["value"][-2][5][3:]
        )

        if "left hand" in cur_data["value"][2]:
            deg = left_max_deg
            dist = left_dist
        elif "right hand" in cur_data["value"][2]:
            deg = right_max_deg
            dist = right_dist
        else:
            deg = (left_max_deg + right_max_deg) / 2
            dist = (left_dist + right_dist) / 2

        if dist > 0.3:
            dist_tag = "long"
        elif dist > 0.15:
            dist_tag = "mid"
        else:
            dist_tag = "short"

        # angular
        if deg > 60:
            rot_tag = "long"
        elif deg > 30:
            rot_tag = "mid"
        else:
            rot_tag = "short"

        static_flag = False
        if (dist_tag == "short" and rot_tag != "long") or (
            rot_tag == "short" and dist_tag != "long"
        ):
            static_flag = True
        else:
            if "grasp" in cur_data["value"][2] or "pick" in cur_data["value"][2]:
                if approach_tag == "no" or approach_tag == "short":
                    static_flag = True
                    continue
            approach_valid[approach_tag].append((cur_idx, cur_data["image"]))
            distance_valid[dist_tag].append((cur_idx, cur_data["image"]))
            rot_valid[rot_tag].append((cur_idx, cur_data["image"]))
            valid_count += 1
            valid_list.append(cur_data)

    return valid_list
