# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 4: Hand Trajectory Extraction (6DoF)

Purpose:
    Extract 6DoF hand trajectories (3D position + quaternion orientation) from source datasets.
    This script processes hand tracking data and aligns it with annotated interaction clips from Step 2.

Dependencies:
    - Runs after: Step 2 (valid_interact_filter.py) - requires filtered interaction annotations
    - Runs before: Step 5 (reason_numeric_qa_generator.py) - provides trajectory data for numeric QA generation

Input:
    - Filtered interaction annotations from Step 2
    - Hand tracking data from source datasets (EgoExo4D, Nymeria):
        * MPS (Machine Perception Services) hand tracking results
        * VRS (Virtual Reality Storage) video recordings
        * Camera calibration parameters
    - For EgoExo4D: Hand tracking data should be located at:
        data/egoman_dataset/egoexo/vrs_list/[take_name]/hand_tracking/

Output:
    - 6DoF hand trajectories aligned with interaction clips:
        * Position: (x, y, z) in meters, camera-relative coordinates
        * Orientation: Quaternion (qx, qy, qz, qw)
        * Frequency: 10 FPS
        * Smooth interpolation for missing frames
    - Projected 2D wrist positions in image coordinates
    - Head pose trajectory for context
    - Camera transformation matrices

Trajectory Format:
    - wrist_pose: Nx6 array (left hand xyz, right hand xyz)
    - wrist_quat: Nx8 array (left hand quaternion xyzw, right hand quaternion xyzw)
    - joint_angles: Nimble hand pose parameters (for EgoExo4D only)
    - project_left_wrist/project_right_wrist: 2D image coordinates

Processing Steps:
    1. Load MPS hand tracking data and VRS recordings
    2. Extract raw 6DoF poses at 10 FPS
    3. Transform from world coordinates to camera-relative coordinates
    4. Apply smoothing to handle tracking noise and missing frames
    5. Align quaternions to canonical hand coordinate frames
    6. Project 3D positions to 2D image coordinates
    7. Save processed trajectories with metadata

Usage:
    python step4_6dof_traj_process.py

    Configure data paths in main():
    - all_data_root: Root directory for EgoMAN dataset
    - egoexo_hand_data_root: EgoExo4D hand tracking directory
    - nymeria_root: Nymeria dataset directory
    - new_anno_dir: Output directory for processed annotations
"""

import base64
import bisect
import json
import logging
import math
import os
import pickle
import random
import re
import ssl
import time
from typing import Dict, final, List, Optional, Union

import cv2
import h5py
import httpx
import jsonlines
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import pytorch3d.transforms as pt
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm


def rotate_90_clockwise(x, y, width, height):
    # Flip vertical
    y = height - 1 - y
    # Flip horizontal
    x = x
    # Transpose
    return y, x


def get_nearest_frame(frames, query_timestamp_ns):
    """
    Find the frame nearest to a given timestamp using binary search (bisect).

    Args:
        frames: list of dicts, each with key 'tracking_timestamp_us'
        query_timestamp_ns: int, timestamp in nanoseconds

    Returns:
        nearest_frame: dict, the frame nearest in time
        nearest_index: int, index of that frame in the list
    """
    if not frames:
        return None, None

    # Convert query timestamp to microseconds to match frames
    query_us = query_timestamp_ns // 1000

    # Extract timestamps (assuming frames sorted by time)
    timestamps = [f["tracking_timestamp_us"] for f in frames]

    # Bisect to find insertion point
    idx = bisect.bisect_left(timestamps, query_us)

    if idx == 0:
        return frames[0], 0
    if idx == len(frames):
        return frames[-1], len(frames) - 1

    # Compare neighbors
    before, after = timestamps[idx - 1], timestamps[idx]
    if abs(query_us - before) <= abs(after - query_us):
        return frames[idx - 1], idx - 1
    else:
        return frames[idx], idx


def transform_pose(T_AB_4x4, p_B_1x3, q_B_xyzw):
    """
    Transform a pose from frame B to frame A using 4x4 T_AB.

    Args:
        T_AB_4x4: 4x4 ndarray, transform from B->A
        p_B_1x3: (3,) or (1,3) position of the wrist in frame B
        q_B_xyzw: (4,) quaternion of the wrist in frame B, format [x, y, z, w]

    Returns:
        p_A_1x3: (3,) position in frame A
        q_A_xyzw: (4,) quaternion in frame A, [x, y, z, w]
    """
    p_B = np.asarray(p_B_1x3).reshape(1, 3)
    p_A, _ = transform_3d_points(T_AB_4x4, p_B)  # position

    # Ensure input quaternion is valid (normalized)
    q_B_norm = np.linalg.norm(q_B_xyzw)
    if q_B_norm == 0:
        raise ValueError("Input quaternion has zero norm")

    # rotation: q_A = q_AB * q_B
    R_AB = R.from_matrix(T_AB_4x4[:3, :3])
    R_B = R.from_quat(q_B_xyzw / q_B_norm)  # normalize input quaternion
    R_A = R_AB * R_B
    q_A_xyzw = R_A.as_quat()

    return p_A.reshape(3), q_A_xyzw


def transform_3d_points(transform, points, gravity_world=None):
    """
    Transform 3D points using a transformation matrix.

    Args:
        transform: 4x4 transformation matrix
        points: Nx3 array of 3D points
        gravity_world: optional gravity vector for alignment

    Returns:
        transformed_points: Nx3 array of transformed points
        final_transform: 4x4 final transformation matrix
    """
    N = len(points)
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed_points_h = (transform @ points_h.T).T
    transformed_points = transformed_points_h[:, :-1]

    final_transform = transform.copy()

    if gravity_world is not None:
        R_world2cam = transform[:3, :3]
        gravity_cam = R_world2cam @ gravity_world
        g_cam_norm = gravity_cam / np.linalg.norm(gravity_cam)

        target_gravity = np.array([0, 0, -1])
        rot_align = R.align_vectors([target_gravity], [g_cam_norm])[0]
        R_align = rot_align.as_matrix()

        # Apply alignment rotation to points
        transformed_points = (R_align @ transformed_points.T).T

        # Compose the final projection matrix with alignment
        final_transform[:3, :3] = R_align @ final_transform[:3, :3]

    return transformed_points, final_transform


def smooth_trajectory_naive(
    main_dict,
    start_id,
    end_id,
    window_size=20,  # must be odd
):
    """Smooth trajectory data using window-based interpolation."""
    filled_dict = {}
    source_flag_dict = {}

    all_times = sorted(set(main_dict.keys()))
    T = max(all_times)

    interp_count = 0
    missing_count = 0

    half_window = window_size // 2

    for t in range(T + 1):
        # Case 1: Use original detection if available
        if (
            t in main_dict
            and isinstance(main_dict[t], np.ndarray)
            and main_dict[t].size == 3
        ):
            filled_dict[t] = main_dict[t]
            source_flag_dict[t] = "detected"
            continue

        # Case 2: Try interpolation using window smoothing
        neighbor_vals = []
        for dt in range(-half_window, half_window + 1):
            tt = t + dt
            if tt in main_dict and isinstance(main_dict[tt], np.ndarray):
                neighbor_vals.append(main_dict[tt])

        if len(neighbor_vals) >= 2:  # enough data to interpolate
            interp_val = np.mean(neighbor_vals, axis=0)
            filled_dict[t] = interp_val
            source_flag_dict[t] = "missing_interp"
            if start_id <= t <= end_id:
                interp_count += 1
            continue

        # Case 3: No enough neighbors, mark as missing
        filled_dict[t] = None
        source_flag_dict[t] = "missing"
        if start_id <= t <= end_id:
            missing_count += 1

    return filled_dict, source_flag_dict, interp_count, missing_count


def get_global_timespan_ns(vrs_provider):
    t_start = vrs_provider.get_first_time_ns_all_streams(TimeDomain.TIME_CODE)
    t_end = vrs_provider.get_last_time_ns_all_streams(TimeDomain.TIME_CODE)
    return t_start, t_end


def get_timespan_ns(vrs_provider_list, ignore_ns: int = 1e9):
    t_start = 0
    t_end = None

    for vrs_provider in vrs_provider_list:
        t0, t1 = get_global_timespan_ns(vrs_provider)
        t_start = t_start if t_start > t0 else t0
        t_end = t_end if t_end is not None and t_end < t1 else t1

    t_start += ignore_ns
    t_end -= ignore_ns
    assert t_start < t_end, f"invalid time span {t_start= }us, {t_end= }us"

    t_start = int(t_start)
    t_end = int(t_end)
    duration = (t_end - t_start) / 1.0e9
    return t_start, t_end, duration


def geometric_mean_quaternions(quaternions, weights=None):
    """
    Compute geometric mean of quaternions using the proper Riemannian metric.
    """
    if not quaternions:
        raise ValueError("Empty quaternion list")

    if weights is None:
        weights = [1.0 / len(quaternions)] * len(quaternions)

    # Convert to rotation matrices for proper averaging
    rotations = [R.from_quat(q) for q in quaternions]

    # Use iterative algorithm to find geometric mean
    # Start with the first quaternion as initial guess
    mean_rot = rotations[0]

    for _ in range(10):  # Maximum iterations
        # Compute weighted sum of log mappings
        log_sum = np.zeros(3)
        for rot, weight in zip(rotations, weights):
            # Compute relative rotation
            rel_rot = mean_rot.inv() * rot
            # Map to tangent space (axis-angle representation)
            axis_angle = rel_rot.as_rotvec()
            log_sum += weight * axis_angle

        # Update mean
        if np.linalg.norm(log_sum) < 1e-6:  # Convergence criterion
            break

        update_rot = R.from_rotvec(log_sum)
        mean_rot = mean_rot * update_rot

    return mean_rot.as_quat()


def smooth_quaternions(quat_dict, start_id, end_id, window_size=20):
    """
    Properly smooth and interpolate quaternions using SLERP and geometric mean.
    """
    filled_dict = {}
    source_flag_dict = {}

    all_times = sorted(set(quat_dict.keys()))
    if not all_times:
        return filled_dict, source_flag_dict, 0, 0

    T = max(all_times)

    interp_count = 0
    missing_count = 0

    half_window = window_size // 2

    for t in range(T + 1):
        # Case 1: Keep original quaternion if available at this exact time t
        if (
            t in quat_dict
            and isinstance(quat_dict[t], np.ndarray)
            and quat_dict[t].size == 4
        ):
            # Validate and normalize the original quaternion
            q_orig = quat_dict[t]
            q_norm = np.linalg.norm(q_orig)
            if q_norm > 1e-8:  # Valid non-zero quaternion
                filled_dict[t] = q_orig / q_norm  # Normalize
                source_flag_dict[t] = "detected"
            else:
                # Zero quaternion, mark as missing
                filled_dict[t] = None
                source_flag_dict[t] = "missing"
                if start_id <= t <= end_id:
                    missing_count += 1
            continue

        # Case 2: Only interpolate if no quaternion exists at time t
        # Look for neighboring quaternions to interpolate from
        neighbor_vals = []
        for dt in range(-half_window, half_window + 1):
            tt = t + dt
            if tt != t and tt in quat_dict and isinstance(quat_dict[tt], np.ndarray):
                if quat_dict[tt].size == 4:  # Ensure it's a valid quaternion shape
                    neighbor_vals.append(quat_dict[tt])

        if len(neighbor_vals) >= 2:  # enough data to interpolate
            try:
                # Validate and normalize quaternions before conversion
                valid_quats = []
                for q in neighbor_vals:
                    if not isinstance(q, np.ndarray) or q.size != 4:
                        continue
                    q = np.asarray(q, dtype=np.float64)  # Ensure proper dtype
                    q_norm = np.linalg.norm(q)
                    if q_norm > 1e-8:  # Valid non-zero quaternion
                        valid_quats.append(q / q_norm)

                if len(valid_quats) < 2:
                    # Not enough valid quaternions
                    filled_dict[t] = None
                    source_flag_dict[t] = "missing"
                    if start_id <= t <= end_id:
                        missing_count += 1
                    continue

                # Convert to Rotation objects with error handling
                rotations = []
                for q in valid_quats:
                    try:
                        rot = R.from_quat(q)
                        rotations.append(rot)
                    except Exception as e:
                        print(
                            f"Warning: Failed to create rotation from quaternion {q}: {e}"
                        )
                        continue

                if len(rotations) < 2:
                    filled_dict[t] = None
                    source_flag_dict[t] = "missing"
                    if start_id <= t <= end_id:
                        missing_count += 1
                    continue

                # Use proper quaternion interpolation/averaging
                if len(rotations) == 2:
                    # For exactly 2 quaternions, we can use SLERP
                    # Since we don't have exact timestamps, use midpoint interpolation
                    times = np.array([0.0, 1.0])
                    slerp = Slerp(times, R.concatenate(rotations))
                    mean_rotation = slerp(0.5)  # Midpoint interpolation
                elif len(rotations) > 2:
                    # For multiple quaternions, use geometric mean
                    all_quats = [rot.as_quat() for rot in rotations]

                    # Use distance-based weighting (closer to center of window gets higher weight)
                    # For simplicity, use equal weights for now
                    mean_quat = geometric_mean_quaternions(all_quats)
                    mean_rotation = R.from_quat(mean_quat)
                else:
                    # Fallback: use the first quaternion if averaging fails
                    mean_rotation = rotations[0]

                result_quat = mean_rotation.as_quat()
                # Final validation
                if np.linalg.norm(result_quat) > 1e-8:
                    filled_dict[t] = result_quat
                    source_flag_dict[t] = "missing_interp"
                    if start_id <= t <= end_id:
                        interp_count += 1
                else:
                    filled_dict[t] = None
                    source_flag_dict[t] = "missing"
                    if start_id <= t <= end_id:
                        missing_count += 1
                continue

            except Exception as e:
                # If rotation interpolation fails, mark as missing
                print(f"Warning: Quaternion interpolation failed at t={t}: {e}")
                filled_dict[t] = None
                source_flag_dict[t] = "missing"
                if start_id <= t <= end_id:
                    missing_count += 1
                continue

        # Case 3: No enough neighbors, mark as missing
        filled_dict[t] = None
        source_flag_dict[t] = "missing"
        if start_id <= t <= end_id:
            missing_count += 1

    return filled_dict, source_flag_dict, interp_count, missing_count


def smooth_joint_angles(joint_dict, start_id, end_id, window_size=20):
    """
    Smooth joint angles using mean interpolation.
    """
    filled_dict = {}
    source_flag_dict = {}

    all_times = sorted(set(joint_dict.keys()))
    T = max(all_times)

    interp_count = 0
    missing_count = 0

    half_window = window_size // 2

    for t in range(T + 1):
        # Case 1: Use original detection if available
        if (
            t in joint_dict
            and isinstance(joint_dict[t], np.ndarray)
            and joint_dict[t].size > 0
        ):
            filled_dict[t] = joint_dict[t]
            source_flag_dict[t] = "detected"
            continue

        # Case 2: Try interpolation using window smoothing
        neighbor_vals = []
        for dt in range(-half_window, half_window + 1):
            tt = t + dt
            if tt in joint_dict and isinstance(joint_dict[tt], np.ndarray):
                neighbor_vals.append(joint_dict[tt])

        if len(neighbor_vals) >= 2:  # enough data to interpolate
            interp_val = np.mean(neighbor_vals, axis=0)
            filled_dict[t] = interp_val
            source_flag_dict[t] = "missing_interp"
            if start_id <= t <= end_id:
                interp_count += 1
            continue

        # Case 3: No enough neighbors, mark as missing
        filled_dict[t] = None
        source_flag_dict[t] = "missing"
        if start_id <= t <= end_id:
            missing_count += 1

    return filled_dict, source_flag_dict, interp_count, missing_count


def align_quat_to_styleA(q_xyzw: np.ndarray, hand: str) -> np.ndarray:
    S_LEFT = np.array(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ]
    )
    S_RIGHT = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    Rm = R.from_quat(q_xyzw).as_matrix()
    S = S_LEFT if hand.lower().startswith("l") else S_RIGHT

    R_aligned = Rm @ S

    fix_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)

    R_final = R_aligned @ fix_matrix
    return R.from_matrix(R_final).as_quat()


def extract_pose_data_egoexo(
    cur_anno,
    take_dict,
    hand_data_root,
    mps_data_provider_dict,
    take_name_hand_dict,
    video_dir,
):
    """Extract pose data for EgoExo dataset following the nimble pipeline."""
    take_uid = cur_anno["take_uid"]
    cur_take = take_dict[take_uid]
    take_name = cur_take["take_info"]["take_name"]
    try:
        aria_cam = [
            cam
            for cam in take_dict[take_uid]["take_info"]["capture"]["cameras"]
            if "aria" in cam["cam_id"]
        ][0]
    except:
        print(f"aria01 not found: {take_name}")
        return None

    # Load MPS data provider
    mps_wrist_csv_path = f"{hand_data_root}/{take_name_hand_dict[take_name]}/hand_tracking/wrist_and_palm_poses.csv"
    if not os.path.exists(mps_wrist_csv_path):
        print(f"hand tracking not found: {take_name}")
        return None

    traj_path = f"{video_dir}/trajectory/closed_loop_trajectory.csv"
    traj_dir = traj_path.split("/trajectory/closed_loop_trajectory.csv")[0]

    if traj_dir not in mps_data_provider_dict:
        vrs_file_name = aria_cam["relative_path"].split("/")[-1]
        vrs_file_path = f"{video_dir}/{vrs_file_name}"
        if not os.path.exists(vrs_file_path):
            vrs_file_name = "A" + vrs_file_name[1:]
            vrs_file_path = f"{video_dir}/{vrs_file_name}"
            if not os.path.exists(vrs_file_path):
                print(f"vrs file not found: {take_name}")
                return None
        provider = data_provider.create_vrs_data_provider(vrs_file_path)
        mps_data_paths_provider = mps.MpsDataPathsProvider(traj_dir)
        mps_data_paths = mps_data_paths_provider.get_data_paths()
        mps_data_provider = mps.MpsDataProvider(mps_data_paths)
        wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(
            mps_wrist_csv_path
        )
        T_device_RGB = provider.get_device_calibration().get_transform_device_sensor(
            "camera-rgb"
        )
        rgb_stream_id = StreamId("214-1")
        rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
        device_calibration = provider.get_device_calibration()
        hand_frames = []
        try:
            with jsonlines.open(
                f"{hand_data_root}/{take_name_hand_dict[take_name]}/hand_tracking/hand_tracking_frames.jsonl"
            ) as reader:
                for frame in reader:
                    hand_frames.append(frame)
        except:
            print(f"hand tracking not found: {take_name}")
            return None
        rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
        mps_data_provider_dict[traj_dir] = (
            mps_data_provider,
            wrist_and_palm_poses,
            T_device_RGB,
            hand_frames,
            device_calibration,
            rgb_camera_calibration,
        )
    else:
        (
            mps_data_provider,
            wrist_and_palm_poses,
            T_device_RGB,
            hand_frames,
            device_calibration,
            rgb_camera_calibration,
        ) = mps_data_provider_dict[traj_dir]

    width, height = rgb_camera_calibration.get_image_size()

    vrs_start_time = take_dict[take_uid]["start_time"]
    start_video_sec = max(0, cur_anno["timestamp"] - 5)

    ref_time_i = float(cur_anno["start_sec"]) + start_video_sec
    end_ref_time_i = float(cur_anno["end_sec"]) + start_video_sec
    T_device_from_camera = T_device_RGB.to_matrix()
    T_camera_from_device = np.linalg.inv(T_device_from_camera)

    # Extract trajectory data
    head_traj = []
    mps_hand_traj = [{}, {}]
    t_count = -1

    start_time = max(0, ref_time_i - 2)
    end_time = cur_anno["end_sec"] + start_video_sec

    for time_i in np.arange(start_time, end_time + 2, 0.1):
        t_count += 1
        time_i_ns = round(time_i * 1e9 + vrs_start_time)
        cur_transform_world_device = mps_data_provider.get_closed_loop_pose(
            time_i_ns
        ).transform_world_device.to_matrix()
        head_pose = cur_transform_world_device[:3, 3]
        head_traj.append(head_pose)

        cur_hand_frame, curidx = get_nearest_frame(hand_frames, time_i_ns)
        cur_wrist_pose = wrist_and_palm_poses[curidx]

        # Left hand
        if cur_hand_frame["hand_poses"]["left"]["existence_confidence"] > 0.0:
            left_wrist_pos = np.array(
                [
                    i / 1e3
                    for i in cur_hand_frame["hand_poses"]["left"]["T_wrist_device"][
                        "translation"
                    ]
                ]
            ).reshape(1, 3)
            left_wrist_quarternion = np.array(
                cur_hand_frame["hand_poses"]["left"]["T_wrist_device"]["quaternion"]
            )
            left_nimble_pose = np.array(
                cur_hand_frame["hand_poses"]["left"]["joint_angles"]
            )
            left_pos_world, left_quat_world = transform_pose(
                cur_transform_world_device, left_wrist_pos, left_wrist_quarternion
            )
            mps_hand_traj[0][t_count] = {
                "wrist_pos": left_pos_world,
                "wrist_quat": left_quat_world,
                "joint_angles": left_nimble_pose,
                "scale": cur_hand_frame["hand_scale"],
            }
        else:
            mps_hand_traj[0][t_count] = None

        # Right hand
        if cur_hand_frame["hand_poses"]["right"]["existence_confidence"] > 0.0:
            right_wrist_pos = np.array(
                [
                    i / 1e3
                    for i in cur_hand_frame["hand_poses"]["right"]["T_wrist_device"][
                        "translation"
                    ]
                ]
            ).reshape(1, 3)
            right_wrist_quarternion = np.array(
                cur_hand_frame["hand_poses"]["right"]["T_wrist_device"]["quaternion"]
            )
            right_nimble_pose = np.array(
                cur_hand_frame["hand_poses"]["right"]["joint_angles"]
            )
            right_pos_world, right_quat_world = transform_pose(
                cur_transform_world_device, right_wrist_pos, right_wrist_quarternion
            )
            mps_hand_traj[1][t_count] = {
                "wrist_pos": right_pos_world,
                "wrist_quat": right_quat_world,
                "joint_angles": right_nimble_pose,
                "scale": cur_hand_frame["hand_scale"],
            }
        else:
            mps_hand_traj[1][t_count] = None

    start_id = math.floor((ref_time_i - start_time) * 10)
    if start_id < 5:
        return None
    else:
        start_id = start_id - 5
    end_id = math.ceil(((end_time - start_time) * 10))

    # Extract separate dictionaries for each component
    left_wrist_pos_dict = {}
    left_wrist_quat_dict = {}
    left_joint_angles_dict = {}
    right_wrist_pos_dict = {}
    right_wrist_quat_dict = {}
    right_joint_angles_dict = {}

    for t_count, hand_data in mps_hand_traj[0].items():
        if hand_data is not None:
            left_wrist_pos_dict[t_count] = hand_data["wrist_pos"]
            left_wrist_quat_dict[t_count] = hand_data["wrist_quat"]
            left_joint_angles_dict[t_count] = hand_data["joint_angles"]
        else:
            left_wrist_pos_dict[t_count] = None
            left_wrist_quat_dict[t_count] = None
            left_joint_angles_dict[t_count] = None

    for t_count, hand_data in mps_hand_traj[1].items():
        if hand_data is not None:
            right_wrist_pos_dict[t_count] = hand_data["wrist_pos"]
            right_wrist_quat_dict[t_count] = hand_data["wrist_quat"]
            right_joint_angles_dict[t_count] = hand_data["joint_angles"]
        else:
            right_wrist_pos_dict[t_count] = None
            right_wrist_quat_dict[t_count] = None
            right_joint_angles_dict[t_count] = None

    # Apply smoothing to each component
    left_wrist_pos_smooth, _, left_pos_interp, left_pos_missing = (
        smooth_trajectory_naive(left_wrist_pos_dict, start_id, end_id)
    )
    left_wrist_quat_smooth, _, left_quat_interp, left_quat_missing = smooth_quaternions(
        left_wrist_quat_dict, start_id, end_id
    )
    left_joint_angles_smooth, _, left_joints_interp, left_joints_missing = (
        smooth_joint_angles(left_joint_angles_dict, start_id, end_id)
    )

    right_wrist_pos_smooth, _, right_pos_interp, right_pos_missing = (
        smooth_trajectory_naive(right_wrist_pos_dict, start_id, end_id)
    )
    right_wrist_quat_smooth, _, right_quat_interp, right_quat_missing = (
        smooth_quaternions(right_wrist_quat_dict, start_id, end_id)
    )
    right_joint_angles_smooth, _, right_joints_interp, right_joints_missing = (
        smooth_joint_angles(right_joint_angles_dict, start_id, end_id)
    )

    # Check if too many frames are missing
    if (
        left_pos_missing > 0
        or right_pos_missing > 0
        or left_quat_missing > 0
        or right_quat_missing > 0
    ):
        return None

    # Extract wrist positions for trajectory processing
    mps_left_wrist_pos = []
    mps_right_wrist_pos = []

    for t in range(start_id, end_id):
        if left_wrist_pos_smooth[t] is not None:
            mps_left_wrist_pos.append(left_wrist_pos_smooth[t])
        if right_wrist_pos_smooth[t] is not None:
            mps_right_wrist_pos.append(right_wrist_pos_smooth[t])

    # Convert to numpy arrays
    mps_left_wrist_pos = np.array(mps_left_wrist_pos)
    mps_right_wrist_pos = np.array(mps_right_wrist_pos)

    if len(mps_left_wrist_pos) == 0 or len(mps_right_wrist_pos) == 0:
        return None

    # Transform data
    ref_time_i_ns = round(ref_time_i * 1e9 + vrs_start_time)
    close_loop_info = mps_data_provider.get_closed_loop_pose(ref_time_i_ns)
    ori_transform_world_device = (
        close_loop_info.transform_world_device.to_matrix().copy()
    )
    transform_world_device = np.linalg.inv(ori_transform_world_device)

    # Transform head trajectory
    head_pose, final_transform = transform_3d_points(
        T_camera_from_device @ transform_world_device, np.array(head_traj)
    )

    # Transform wrist poses (both position and quaternion)
    left_wrist_poses_transformed = []
    right_wrist_poses_transformed = []
    project_left_wrist_list = []
    project_right_wrist_list = []

    # Transform left hand poses using world->camera transformation
    for t in range(start_id, end_id):
        if (
            left_wrist_pos_smooth[t] is not None
            and left_wrist_quat_smooth[t] is not None
        ):
            pos_transformed, quat_transformed = transform_pose(
                T_camera_from_device @ transform_world_device,
                left_wrist_pos_smooth[t],
                left_wrist_quat_smooth[t],
            )
            left_wrist_poses_transformed.append(
                {"pos": pos_transformed, "quat": quat_transformed}
            )
            # Project left wrist to image coordinates using rgb_camera_calibration
            project_left = rgb_camera_calibration.project(pos_transformed)
            if project_left is not None:
                width, height = rgb_camera_calibration.get_image_size()
                x, y = project_left
                x, y = rotate_90_clockwise(x, y, width, height)
                project_left = [x, y]
            project_left_wrist_list.append(project_left)
        else:
            project_left_wrist_list.append(None)

    # Transform right hand poses using world->camera transformation
    for t in range(start_id, end_id):
        if (
            right_wrist_pos_smooth[t] is not None
            and right_wrist_quat_smooth[t] is not None
        ):
            pos_transformed, quat_transformed = transform_pose(
                T_camera_from_device @ transform_world_device,
                right_wrist_pos_smooth[t],
                right_wrist_quat_smooth[t],
            )
            right_wrist_poses_transformed.append(
                {"pos": pos_transformed, "quat": quat_transformed}
            )
            # Project right wrist to image coordinates using rgb_camera_calibration
            project_right = rgb_camera_calibration.project(pos_transformed)
            if project_right is not None:
                width, height = rgb_camera_calibration.get_image_size()
                x, y = project_right
                x, y = rotate_90_clockwise(x, y, width, height)
                project_right = [x, y]
            project_right_wrist_list.append(project_right)
        else:
            project_right_wrist_list.append(None)

    if (
        len(left_wrist_poses_transformed) == 0
        or len(right_wrist_poses_transformed) == 0
    ):
        return None

    # Extract positions for backward compatibility
    mpsleft_wrist_aligned = np.array(
        [pose["pos"] for pose in left_wrist_poses_transformed]
    )
    mpsright_wrist_aligned = np.array(
        [pose["pos"] for pose in right_wrist_poses_transformed]
    )

    # Process current annotations
    head_pose = np.round(head_pose[start_id:end_id], 3).tolist()
    wrist_pose = np.round(
        np.concatenate([mpsleft_wrist_aligned, mpsright_wrist_aligned], axis=1), 3
    ).tolist()

    return {
        "head_pose": head_pose,
        "wrist_pose": wrist_pose,
        "camera_pose": T_device_from_camera.tolist(),
        "transform_world_device": ori_transform_world_device.tolist(),
        "transform": final_transform.tolist(),
        "left_hand_data": {
            "wrist_pos_transformed": [
                pose["pos"] for pose in left_wrist_poses_transformed
            ],
            "wrist_quat_transformed": [
                pose["quat"] for pose in left_wrist_poses_transformed
            ],
            "joint_angles": [
                left_joint_angles_smooth[t]
                for t in range(start_id, end_id)
                if left_joint_angles_smooth[t] is not None
            ],
        },
        "right_hand_data": {
            "wrist_pos_transformed": [
                pose["pos"] for pose in right_wrist_poses_transformed
            ],
            "wrist_quat_transformed": [
                pose["quat"] for pose in right_wrist_poses_transformed
            ],
            "joint_angles": [
                right_joint_angles_smooth[t]
                for t in range(start_id, end_id)
                if right_joint_angles_smooth[t] is not None
            ],
        },
        "project_left_wrist": project_left_wrist_list,
        "project_right_wrist": project_right_wrist_list,
    }


def extract_pose_data_nymeria(cur_anno, mps_data_provider_dict, root):
    """Extract pose data for Nymeria dataset following the nymeria pipeline."""
    video_id = cur_anno["video"].split("/")[-2]
    anno_dir = f"{root}/{video_id}/recording_head/mps/"

    if anno_dir not in mps_data_provider_dict:
        mps_data_paths_provider = mps.MpsDataPathsProvider(anno_dir)
        mps_data_paths = mps_data_paths_provider.get_data_paths()
        mps_data_provider = mps.MpsDataProvider(mps_data_paths)
        provider = data_provider.create_vrs_data_provider(
            f"{root}/{video_id}/recording_head/data/data.vrs"
        )
        video_start_time = 0
        video_last_time = 0
        for stream_id in provider.get_all_streams():
            label = provider.get_label_from_stream_id(stream_id)
            if label == "camera-rgb":
                start_time_ns = provider.get_first_time_ns(
                    stream_id, TimeDomain.DEVICE_TIME
                )
                last_time_ns = provider.get_last_time_ns(
                    stream_id, TimeDomain.DEVICE_TIME
                )
                T_device_RGB = (
                    provider.get_device_calibration().get_transform_device_sensor(label)
                )
                video_start_time = start_time_ns / 1e9
                video_last_time = last_time_ns / 1e9
                break

        lw_folder = f"{root}/{video_id}/recording_lwrist/mps"
        lw_mps_data_paths_provider = mps.MpsDataPathsProvider(lw_folder)
        lw_mps_data_paths = lw_mps_data_paths_provider.get_data_paths()
        lw_mps_data_provider = mps.MpsDataProvider(lw_mps_data_paths)

        rw_folder = f"{root}/{video_id}/recording_rwrist/mps"
        rw_mps_data_paths_provider = mps.MpsDataPathsProvider(rw_folder)
        rw_mps_data_paths = rw_mps_data_paths_provider.get_data_paths()
        rw_mps_data_provider = mps.MpsDataProvider(rw_mps_data_paths)

        lw_provider = data_provider.create_vrs_data_provider(
            f"{root}/{video_id}/recording_lwrist/data/motion.vrs"
        )
        rw_provider = data_provider.create_vrs_data_provider(
            f"{root}/{video_id}/recording_rwrist/data/motion.vrs"
        )
        if rw_provider is None or lw_provider is None:
            return None

        lw_start_time_ns = lw_provider.get_first_time_ns(
            lw_provider.get_all_streams()[0], TimeDomain.DEVICE_TIME
        )
        rw_start_time_ns = rw_provider.get_first_time_ns(
            rw_provider.get_all_streams()[0], TimeDomain.DEVICE_TIME
        )
        gloabl_start_time, global_end_time, global_duration = get_timespan_ns(
            [provider, lw_provider, rw_provider]
        )
        t_ns_device = provider.convert_from_timecode_to_device_time_ns(
            timecode_time_ns=gloabl_start_time
        )
        video_anno_time_shift = (t_ns_device - start_time_ns) / 1e9
        lw_shift_time_ns = (
            lw_provider.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=gloabl_start_time
            )
            - t_ns_device
        )
        rw_shift_time_ns = (
            rw_provider.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=gloabl_start_time
            )
            - t_ns_device
        )
        video_anno_time_shift = (t_ns_device - start_time_ns) / 1e9
        mps_data_provider_dict[anno_dir] = [
            mps_data_provider,
            provider,
            start_time_ns,
            lw_mps_data_provider,
            rw_mps_data_provider,
            video_anno_time_shift,
            t_ns_device,
            lw_shift_time_ns,
            rw_shift_time_ns,
            T_device_RGB,
            stream_id,
            label,
        ]
    else:
        [
            mps_data_provider,
            provider,
            start_time_ns,
            lw_mps_data_provider,
            rw_mps_data_provider,
            video_anno_time_shift,
            t_ns_device,
            lw_shift_time_ns,
            rw_shift_time_ns,
            T_device_RGB,
            stream_id,
            label,
        ] = mps_data_provider_dict[anno_dir]

    vrs_start_time = start_time_ns
    cur_anno["start_time_ns"] = vrs_start_time
    cur_time_stamp = cur_anno["video"].split("/")[-1][:-4].split("_")[-2]
    time_i = (
        float(cur_time_stamp) + cur_anno["start_sec"]
    )  # Use end timestamp for pose extraction
    time_i_ns = round(time_i * 1e9 + vrs_start_time)

    close_loop_info = mps_data_provider.get_closed_loop_pose(time_i_ns)
    ori_transform_world_device = close_loop_info.transform_world_device.to_matrix()
    transform_world_device = np.linalg.inv(ori_transform_world_device)

    T_device_from_camera = T_device_RGB.to_matrix()
    T_camera_from_device = np.linalg.inv(T_device_from_camera)

    fps = 10
    head_pose_list = []
    left_wrist_pose_list = []
    right_wrist_pose_list = []
    if time_i - 0.5 < 0:
        start_time = 0
    start_time = max(time_i - 0.5, 0)
    end_time = float(cur_time_stamp) + cur_anno["end_sec"]
    duration = end_time - start_time
    time_num = round(duration * fps)

    for tid in range(time_num):
        cur_time_ns = round(start_time * 1e9 + tid * 1e9 * 1.0 / fps + vrs_start_time)
        traj = mps_data_provider.get_closed_loop_pose(cur_time_ns)
        trans = traj.transform_world_device.to_quat_and_translation()  # qwqxqyqztxtytz
        velocity = traj.device_linear_velocity_device
        angular = traj.angular_velocity_device
        head_pose = np.hstack(
            [
                trans.reshape(-1),
                velocity.reshape(-1),
                angular.reshape(-1),
            ]
        )
        head_pose_list.append(head_pose)

        lw_traj = lw_mps_data_provider.get_closed_loop_pose(
            cur_time_ns + lw_shift_time_ns
        )
        lw_trans = (
            lw_traj.transform_world_device.to_quat_and_translation()
        )  # qwqxqyqztxtytz
        lw_velocity = lw_traj.device_linear_velocity_device
        lw_angular = lw_traj.angular_velocity_device
        left_wrist_pose = np.hstack(
            [
                lw_trans.reshape(-1),
                lw_velocity.reshape(-1),
                lw_angular.reshape(-1),
            ]
        )
        left_wrist_pose_list.append(left_wrist_pose)

        rw_traj = rw_mps_data_provider.get_closed_loop_pose(
            cur_time_ns + rw_shift_time_ns
        )
        rw_trans = (
            rw_traj.transform_world_device.to_quat_and_translation()
        )  # qwqxqyqztxtytz
        rw_velocity = rw_traj.device_linear_velocity_device
        rw_angular = rw_traj.angular_velocity_device
        right_wrist_pose = np.hstack(
            [
                rw_trans.reshape(-1),
                rw_velocity.reshape(-1),
                rw_angular.reshape(-1),
            ]
        )
        right_wrist_pose_list.append(right_wrist_pose)
    head_pose_list = np.array(head_pose_list)
    left_wrist_pose = np.array(left_wrist_pose_list)
    right_wrist_pose = np.array(right_wrist_pose_list)

    # Initialize lists to store transformed poses
    new_head_pose_list = []
    new_head_quat_list = []
    new_head_vel_list = []
    new_left_wrist_list = []
    new_left_quat_list = []
    new_left_vel_list = []
    new_right_wrist_list = []
    new_right_quat_list = []
    new_right_vel_list = []
    new_project_left_wrist_list = []
    new_project_right_wrist_list = []
    device_calib = provider.get_device_calibration()
    rgb_calib = device_calib.get_camera_calib(label)
    for t in range(len(head_pose_list)):
        # Transform head poses - convert quaternion from wxyz to xyzw format
        head_quat_wxyz = head_pose_list[t, :4]  # w, x, y, z
        head_quat_xyzw = np.array(
            [head_quat_wxyz[1], head_quat_wxyz[2], head_quat_wxyz[3], head_quat_wxyz[0]]
        )  # x, y, z, w
        head_pos = head_pose_list[t, 4:7]  # translation part
        head_vel = head_pose_list[t, 7:10]  # velocity part

        new_head_pose, new_head_quat = transform_pose(
            T_camera_from_device @ transform_world_device, head_pos, head_quat_xyzw
        )
        new_head_pose_list.append(new_head_pose)
        new_head_quat_list.append(new_head_quat)
        new_head_vel_list.append(head_vel)

        # Transform left wrist poses - convert quaternion from wxyz to xyzw format
        left_quat_wxyz = left_wrist_pose[t, :4]  # w, x, y, z
        left_quat_xyzw = np.array(
            [left_quat_wxyz[1], left_quat_wxyz[2], left_quat_wxyz[3], left_quat_wxyz[0]]
        )  # x, y, z, w
        left_pos = left_wrist_pose[t, 4:7]  # translation part
        left_vel = left_wrist_pose[t, 7:10]  # velocity part

        new_left_wrist, new_left_quat = transform_pose(
            T_camera_from_device @ transform_world_device, left_pos, left_quat_xyzw
        )
        width, height = rgb_calib.get_image_size()

        project_left_wrist = rgb_calib.project(new_left_wrist)
        if project_left_wrist is not None:
            x, y = project_left_wrist

            x, y = rotate_90_clockwise(x, y, width, height)
            project_left_wrist = [x, y]
        new_project_left_wrist_list.append(project_left_wrist)

        new_left_wrist_list.append(new_left_wrist)
        new_left_quat_list.append(new_left_quat)
        new_left_vel_list.append(left_vel)

        # Transform right wrist poses - convert quaternion from wxyz to xyzw format
        right_quat_wxyz = right_wrist_pose[t, :4]  # w, x, y, z
        right_quat_xyzw = np.array(
            [
                right_quat_wxyz[1],
                right_quat_wxyz[2],
                right_quat_wxyz[3],
                right_quat_wxyz[0],
            ]
        )  # x, y, z, w
        right_pos = right_wrist_pose[t, 4:7]  # translation part
        right_vel = right_wrist_pose[t, 7:10]  # velocity part

        new_right_wrist, new_right_quat = transform_pose(
            T_camera_from_device @ transform_world_device, right_pos, right_quat_xyzw
        )
        project_right_wrist = rgb_calib.project(new_right_wrist)
        if project_right_wrist is not None:
            x, y = project_right_wrist
            x, y = rotate_90_clockwise(x, y, width, height)
            project_right_wrist = [x, y]
        new_project_right_wrist_list.append(project_right_wrist)

        new_right_wrist_list.append(new_right_wrist)
        new_right_quat_list.append(new_right_quat)
        new_right_vel_list.append(right_vel)

    # rgb_calib.project(new_left_wrist)

    # Convert lists to numpy arrays
    new_head_pose = np.array(new_head_pose_list)
    new_head_quat = np.array(new_head_quat_list)
    new_head_vel = np.array(new_head_vel_list)
    new_left_wrist = np.array(new_left_wrist_list)
    new_left_quat = np.array(new_left_quat_list)
    new_left_vel = np.array(new_left_vel_list)
    new_right_wrist = np.array(new_right_wrist_list)
    new_right_quat = np.array(new_right_quat_list)
    new_right_vel = np.array(new_right_vel_list)
    new_wrist_pose = np.concatenate([new_left_wrist, new_right_wrist], axis=1)
    new_wrist_quat = np.concatenate([new_left_quat, new_right_quat], axis=1)
    new_wrist_vel = np.concatenate([new_left_vel, new_right_vel], axis=1)

    return {
        "start_time_ns": vrs_start_time,
        "t_ns_device": t_ns_device,
        "lw_shift_time_ns": lw_shift_time_ns,
        "rw_shift_time_ns": rw_shift_time_ns,
        "head_pose": np.round(new_head_pose, 3).tolist(),
        "head_quat": np.round(new_head_quat, 6).tolist(),
        "head_velocity": np.round(new_head_vel, 6).tolist(),
        "wrist_pose": np.round(new_wrist_pose, 3).tolist(),
        "wrist_quat": np.round(new_wrist_quat, 6).tolist(),
        "wrist_velocity": np.round(new_wrist_vel, 6).tolist(),
        "transform_world_device": ori_transform_world_device.tolist(),
        "gravity_world": close_loop_info.gravity_world.tolist(),
        "project_left_wrist": new_project_left_wrist_list,
        "project_right_wrist": new_project_right_wrist_list,
    }


if __name__ == "__main__":
    all_data_root = "../data/egoman_dataset"
    all_list = pickle.load(open(f"{all_data_root}/valid_egoman_anno_forhand.pkl", "rb"))

    all_anno = []
    valid_annos = []

    # Load dataset configurations
    egoexo_take_dict = None
    egoexo_hand_data_root = (
        f"{all_data_root}/egoexo/vrs_list"  # all hand tracking data (for joint angles)
    )
    nymeria_root = f"{all_data_root}/nymeria_dataset"
    new_anno_dir = f"{all_data_root}/processed_egoman_anno"

    # Load take name mapping for egoexo
    take_name_hand_dict = {}
    if os.path.exists(egoexo_hand_data_root):
        for take_name_ori in os.listdir(egoexo_hand_data_root):
            take_name = take_name_ori.split("_aria")[0]
            take_name_hand_dict[take_name] = take_name_ori

    mps_data_provider_dict = {}

    random.shuffle(all_list)
    os.makedirs(new_anno_dir, exist_ok=True)
    for cur_anno in tqdm(all_list):

        dataset = cur_anno["dataset"]

        video_id = cur_anno["video"].split("/")[-2]
        video_sec = round(float(cur_anno["timestamp"]), 2)
        output_file_path = os.path.join(
            new_anno_dir,
            f"{video_id}_videostamp_{video_sec}_timestamp_{cur_anno['start_sec']}_{cur_anno['end_sec']}.pkl",
        )

        pose_data = None

        if dataset == "egoexo":
            take_uid = cur_anno["take_uid"]
            # Load take dict if not already loaded
            if egoexo_take_dict is None:
                take_dict_path = f"{all_data_root}/egoexo/vli_takes_info_byuid.json"
                if os.path.exists(take_dict_path):
                    egoexo_take_dict = json.load(open(take_dict_path, "r"))
                else:
                    print(f"EgoExo take dict not found at {take_dict_path}")
                    continue

            aria_cam = [
                cam
                for cam in egoexo_take_dict[take_uid]["take_info"]["capture"]["cameras"]
                if "aria" in cam["cam_id"]
            ][0]
            if aria_cam == "aria01":
                if os.path.exists(output_file_path):
                    continue
            else:
                if (
                    os.path.exists(output_file_path)
                    and len(pickle.load(open(output_file_path, "rb"))) > 0
                ):
                    continue

            pose_data = extract_pose_data_egoexo(
                cur_anno,
                egoexo_take_dict,
                egoexo_hand_data_root,
                mps_data_provider_dict,
                take_name_hand_dict,
                video_dir=f"{all_data_root}/egoexo/takes/{take_name}",
            )

            left_quat = np.array(pose_data["left_hand_data"]["wrist_quat_transformed"])
            right_quat = np.array(
                pose_data["right_hand_data"]["wrist_quat_transformed"]
            )
            left_quat = align_quat_to_styleA(left_quat, "left")
            right_quat = align_quat_to_styleA(right_quat, "right")
            pose_data["left_hand_data"]["wrist_quat_transformed"] = left_quat
            pose_data["right_hand_data"]["wrist_quat_transformed"] = right_quat

            video_name = cur_anno["video"][:-4]
            video_time = round(float(video_name.split("/")[-1]), 2)
            cur_anno["video"] = (
                video_name.replace(video_name.split("/")[-1], str(video_time)) + ".mp4"
            )

        elif dataset == "nymeria":
            if os.path.exists(output_file_path):
                continue
            pose_data = extract_pose_data_nymeria(
                cur_anno, mps_data_provider_dict, nymeria_root
            )

        if pose_data is not None:
            # Merge pose data into current annotation
            cur_anno["pose_data"] = pose_data
            pickle.dump(cur_anno, open(output_file_path, "wb"))
        else:
            pickle.dump({}, open(output_file_path, "wb"))
