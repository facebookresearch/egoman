# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
put this file under github hot3d folder: https://github.com/facebookresearch/hot3d/tree/main/hot3d
"""
"""
Step 4: Hand Trajectory Extraction (6DoF) for HOT3D Dataset (Benchmark Evaluation Only)

Purpose:
    Extract 6DoF hand trajectories (3D position + quaternion orientation) from the HOT3D dataset.
    This is a specialized version of step4_6dof_traj_process.py specifically for HOT3D data format.

    This script:
    1. Detects object movement timeranges from dynamic object tracking data
    2. Extracts hand poses from UMETrack hand tracking JSONL files
    3. Aligns hand trajectories with object interaction periods
    4. Transforms world-frame poses to camera-relative coordinates
    5. Projects 3D hand positions to 2D image coordinates

Dependencies:
    - Runs after: Step 2 (valid_interact_filter.py) - requires filtered interaction annotations
    - Requires: HOT3D repository (https://github.com/facebookresearch/hot3d)

Input:
    - HOT3D dataset structure:
        * dynamic_objects.csv: Object 6DoF poses (position + quaternion)
        * umetrack_hand_pose_trajectory.jsonl: Hand tracking data
        * recording.vrs: Aria VRS video recordings
        * mps/: Machine Perception Services data (poses, calibration)
        * camera_models.json: Camera intrinsics/extrinsics
    - Object library with 3D models
    - Metadata with object names and UIDs

Output:
    - 6DoF hand trajectories aligned with object interaction clips:
        * Position: (x, y, z) in meters, camera-relative coordinates
        * Orientation: Quaternion (x, y, z, w)
        * Frequency: 10 FPS
        * Time range: 0.5s before interaction to end + buffer
    - Object trajectories in camera frame
    - 2D wrist projections in image coordinates
    - Camera pose and transform matrices

Processing Workflow:
    1. **Object Movement Detection:**
       - Analyze object pose changes to detect movement timeranges
       - Merge close timeranges to form continuous interaction periods
       - Filter by minimum duration (0.1s)

    2. **Hand Trajectory Extraction:**
       - Load UMETrack hand poses from JSONL (left/right wrist transforms)
       - Sample at 10 FPS aligned with object interaction periods
       - Convert quaternions from wxyz to xyzw format

    3. **Coordinate Transformation:**
       - Transform from world frame to camera-relative frame at query time
       - Apply device-to-sensor (RGB camera) calibration
       - Ensure consistent coordinate system across modalities

    4. **2D Projection:**
       - Project 3D hand wrists to 2D image coordinates
       - Rotate image 90° clockwise to match Aria coordinate convention
       - Validate projections are within image bounds

Usage:
    Place this file under: hot3d/hot3d/step4_6dof_traj_process_hot3d.py

    Configure paths in __main__:
    - HOT3D_ROOT: Root directory of HOT3D dataset
    - IMG_OUT_DIR: Output directory for extracted images
    - ANNO_PKL_OUT: Output path for processed annotations

    Run: python step4_6dof_traj_process_hot3d.py

Output Format:
    Pickle file with list of interaction records, each containing:
    - times: Timestamps at 10 FPS
    - left_hand_traj: (T, 7) [x, y, z, qx, qy, qz, qw]
    - right_hand_traj: (T, 7) [x, y, z, qx, qy, qz, qw]
    - object_traj: (T, 7) object pose in camera frame
    - manipulating_hand: "left" or "right"
    - camera_T_world_at_question: SE3 transform matrix
    - image: Path to saved reference frame image

Note: This script requires HOT3D dataset and the HOT3D repository to be set up.
      Install dependencies: projectaria_tools, pytorch3d, and HOT3D data loaders.
"""

import io
import json
import os
import pickle
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Todo move up later
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.calibration import CameraCalibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sophus import SE3
from tqdm import tqdm


from data_loaders.HeadsetPose3dProvider import HeadsetPose3dProvider
from data_loaders.loader_hand_poses import Handedness, HandPose3dCollection
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from dataset_api import Hot3dDataProvider



# Option A: you know the exact stream id (e.g., slam-left fisheye)
stream_id = StreamId("214-1")  # <- change to your stream
image_streamid = StreamId("214-1")
# If you want to use a fisheye/slam stream (like the tutorial), set USE_STREAM_ID=True and STREAM_ID_STR accordingly.
# Otherwise, leave USE_STREAM_ID=False and set IMAGE_STREAM_LABEL = "camera-rgb" to use RGB.
USE_STREAM_ID = False
STREAM_ID_STR = "214-1"  # e.g., slam-left; used only if USE_STREAM_ID = True
IMAGE_STREAM_LABEL = "camera-rgb"  # used only if USE_STREAM_ID = False



def calculate_position_change(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos2 - pos1)


def calculate_rotation_change(q1: np.ndarray, q2: np.ndarray) -> float:
    """Calculate angular difference between two quaternions in radians."""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate dot product and clamp to valid range
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Return angular difference
    return 2 * np.arccos(dot_product)


def merge_close_timeranges(
    timeranges: List[Dict[str, Any]],
    max_gap_seconds: float = 2.0,
    min_merged_duration: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Merge timeranges that are close to each other to create larger interaction periods.

    Args:
        timeranges: List of movement timeranges
        max_gap_seconds: Maximum gap between timeranges to merge (seconds)
        min_merged_duration: Minimum duration for merged timeranges (seconds)

    Returns:
        List of merged timeranges
    """
    if not timeranges:
        return []

    # Sort by start timestamp
    sorted_ranges = sorted(timeranges, key=lambda x: x["start_timestamp_ns"])
    merged_ranges = []
    current_range = sorted_ranges[0].copy()

    for next_range in sorted_ranges[1:]:
        gap_ns = next_range["start_timestamp_ns"] - current_range["end_timestamp_ns"]
        gap_seconds = gap_ns / 1e9

        if gap_seconds <= max_gap_seconds:
            # Merge ranges
            current_range["end_timestamp_ns"] = next_range["end_timestamp_ns"]
            current_range["duration_ns"] = (
                current_range["end_timestamp_ns"] - current_range["start_timestamp_ns"]
            )
            current_range["duration_seconds"] = current_range["duration_ns"] / 1e9
            current_range["frame_count"] += next_range["frame_count"]

            # Update statistics (take maximum values for merged periods)
            current_range["max_position_change_m"] = max(
                current_range["max_position_change_m"],
                next_range["max_position_change_m"],
            )
            current_range["max_rotation_change_rad"] = max(
                current_range["max_rotation_change_rad"],
                next_range["max_rotation_change_rad"],
            )

            # Average the average values weighted by frame count
            total_frames = current_range["frame_count"]
            current_avg_pos = current_range["avg_position_change_m"]
            next_avg_pos = next_range["avg_position_change_m"]
            current_range["avg_position_change_m"] = (
                current_avg_pos * (total_frames - next_range["frame_count"])
                + next_avg_pos * next_range["frame_count"]
            ) / total_frames

            current_avg_rot = current_range["avg_rotation_change_rad"]
            next_avg_rot = next_range["avg_rotation_change_rad"]
            current_range["avg_rotation_change_rad"] = (
                current_avg_rot * (total_frames - next_range["frame_count"])
                + next_avg_rot * next_range["frame_count"]
            ) / total_frames

            # Mark as merged
            current_range["is_merged"] = True
            current_range["merged_from_count"] = (
                current_range.get("merged_from_count", 1) + 1
            )
        else:
            # Gap too large, finalize current range
            if current_range["duration_seconds"] >= min_merged_duration:
                merged_ranges.append(current_range)
            current_range = next_range.copy()

    # Add the last range
    if current_range["duration_seconds"] >= min_merged_duration:
        merged_ranges.append(current_range)

    return merged_ranges

    # Sort interaction periods by start time
    interaction_periods.sort(key=lambda x: x["start_timestamp_ns"])

    return interaction_periods


def detect_movement_timeranges(
    object_data: pd.DataFrame,
    position_threshold: float = 0.01,  # 1cm
    rotation_threshold: float = 0.1,  # ~6 degrees
    min_frames: int = 5,
    gap_tolerance: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detect timeranges where object is moving based on pose changes.

    Args:
        object_data: DataFrame containing pose data for a single object
        position_threshold: Minimum position change (meters) to consider as movement
        rotation_threshold: Minimum rotation change (radians) to consider as movement
        min_frames: Minimum number of consecutive frames to consider as valid movement
        gap_tolerance: Number of non-moving frames to tolerate within a movement sequence

    Returns:
        List of movement timeranges with start/end timestamps and statistics
    """
    if len(object_data) < 2:
        return []

    # Sort by timestamp
    data = object_data.sort_values("timestamp[ns]").copy()

    # Calculate pose changes between consecutive frames
    position_cols = ["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]
    rotation_cols = ["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]

    position_changes = []
    rotation_changes = []

    for i in range(1, len(data)):
        pos1 = data.iloc[i - 1][position_cols].values
        pos2 = data.iloc[i][position_cols].values

        rot1 = data.iloc[i - 1][rotation_cols].values
        rot2 = data.iloc[i][rotation_cols].values

        pos_change = calculate_position_change(pos1, pos2)
        rot_change = calculate_rotation_change(rot1, rot2)

        position_changes.append(pos_change)
        rotation_changes.append(rot_change)

    # Determine movement frames (skip first frame as we don't have previous data)
    is_moving = []
    for i in range(len(position_changes)):
        moving = (
            position_changes[i] > position_threshold
            or rotation_changes[i] > rotation_threshold
        )
        is_moving.append(moving)

    # Find continuous movement periods with gap tolerance
    movement_periods = []
    current_start = None
    gap_count = 0

    for i, moving in enumerate(is_moving):
        if moving:
            if current_start is None:
                current_start = i + 1  # +1 because is_moving is offset by 1
            gap_count = 0
        else:
            if current_start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    # End current movement period
                    end_idx = i - gap_count + 1
                    if end_idx - current_start >= min_frames:
                        movement_periods.append((current_start, end_idx))
                    current_start = None
                    gap_count = 0

    # Handle case where movement continues until end
    if current_start is not None:
        end_idx = len(is_moving)
        if end_idx - current_start >= min_frames:
            movement_periods.append((current_start, end_idx))

    # Convert to timeranges with statistics
    timeranges = []
    for start_idx, end_idx in movement_periods:
        start_timestamp = int(data.iloc[start_idx]["timestamp[ns]"])
        end_timestamp = int(data.iloc[end_idx]["timestamp[ns]"])

        # Calculate movement statistics for this period
        period_pos_changes = position_changes[start_idx - 1 : end_idx - 1]
        period_rot_changes = rotation_changes[start_idx - 1 : end_idx - 1]

        timerange = {
            "start_timestamp_ns": start_timestamp,
            "end_timestamp_ns": end_timestamp,
            "duration_ns": end_timestamp - start_timestamp,
            "duration_seconds": (end_timestamp - start_timestamp) / 1e9,
            "frame_count": end_idx - start_idx + 1,
            "max_position_change_m": (
                max(period_pos_changes) if period_pos_changes else 0
            ),
            "avg_position_change_m": (
                np.mean(period_pos_changes) if period_pos_changes else 0
            ),
            "max_rotation_change_rad": (
                max(period_rot_changes) if period_rot_changes else 0
            ),
            "avg_rotation_change_rad": (
                np.mean(period_rot_changes) if period_rot_changes else 0
            ),
        }
        timeranges.append(timerange)

    return timeranges


def analyze_dynamic_objects(csv_path: str, metadata_path: str) -> Dict[str, Any]:
    """
    Analyze dynamic objects and detect movement timeranges.

    Args:
        csv_path: Path to dynamic_objects.csv
        metadata_path: Path to metadata.json

    Returns:
        Dictionary containing analysis results for all objects
    """
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Create object name to UID mapping
    object_mapping = {}
    for name, uid in zip(metadata["object_names"], metadata["object_uids"]):
        object_mapping[name] = uid

    # Load dynamic objects data
    df = pd.read_csv(csv_path)

    # Convert timestamp to int for consistency
    df["timestamp[ns]"] = df["timestamp[ns]"].astype(int)

    # Analyze each object - simplified output
    results = {
        "metadata": {
            "participant_id": metadata["participant_id"],
            "recording_name": metadata["recording_name"],
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            # "first_timestamp_ns": int(first_timestamp_ns),
        },
        "objects": {},
    }
    valid_count = 0
    for object_name, object_uid in object_mapping.items():
        # print(f"Analyzing {object_name}...")

        # Filter data for this object
        object_data = df[df["object_uid"] == int(object_uid)].copy()

        if len(object_data) == 0:
            #     print(f"  No data found")
            continue

        # Detect movement timeranges
        movement_timeranges = detect_movement_timeranges(object_data)

        filtered_timeranges = [
            tr for tr in movement_timeranges if tr["duration_seconds"] >= 0.1
        ]

        merged_timeranges = merge_close_timeranges(
            filtered_timeranges, max_gap_seconds=0.1, min_merged_duration=0.1
        )

        # Convert timestamps to seconds relative to first timestamp
        relative_segments = []
        for seg in merged_timeranges:
            relative_seg = {
                "start_time_s": (seg["start_timestamp_ns"]) / 1e9,
                "end_time_s": (seg["end_timestamp_ns"]) / 1e9,
                "duration_s": seg["duration_seconds"],
                "frame_count": seg["frame_count"],
            }
            relative_segments.append(relative_seg)

        # Store only the essential information
        results["objects"][object_name] = {
            "object_uid": object_uid,
            "total_segments": len(relative_segments),
            "segments": relative_segments,
        }
        valid_count += 1
    if valid_count == 0:
        results["objects"] = {}
    return results


def load_timecode_device_mapping(csv_path: str) -> Dict[int, int]:
    """Load timecode→device_time mapping from the CSV under MPS folder."""
    import csv

    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tc = int(row["timecode_ns"])
            dev = int(row["devicetime_ns"])
            mapping[tc] = dev
    return mapping


def timecode_to_devicetime(timecode_ns, mapping_dict):
    """Map timecode_ns to devicetime_ns using the closest match."""
    timecodes = list(mapping_dict.keys())
    idx = min(range(len(timecodes)), key=lambda i: abs(timecodes[i] - timecode_ns))
    return mapping_dict[timecodes[idx]]


# ---------------------------
# Math helpers
# ---------------------------
def quat_normalize_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_xyzw_to_R(q):
    x, y, z, w = quat_normalize_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def R_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            y = 0.25 * s
            x = (R[0, 1] + R[1, 0]) / s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            z = 0.25 * s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            w = (R[1, 0] - R[0, 1]) / s
    return quat_normalize_xyzw(np.array([x, y, z, w], dtype=np.float64))


def find_closest_time(
    time_dict: Dict[float, Any], query_time: float
) -> Tuple[float, Any]:
    times = list(time_dict.keys())
    idx = min(range(len(times)), key=lambda i: abs(times[i] - query_time))
    closest_time = times[idx]
    return closest_time, time_dict[closest_time]


def retrieve_device_pose(
    timestamp_ns: int,
    stream_id: StreamId,
    device_pose_provider: Optional[HeadsetPose3dProvider] = None,
    device_data_provider: Optional[Any] = None,
) -> Optional[tuple[SE3, CameraCalibration]]:
    """
    Retrieve the pose of the device and apply the device_camera transformation on top of it for the provided stream_id
    """
    headset_pose3d_with_dt = None
    if device_pose_provider is not None:
        headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )

        if headset_pose3d_with_dt is not None:
            headset_pose3d = headset_pose3d_with_dt.pose3d

            # Retrieve the camera calibration (intrinsics and extrinsics) for a given stream_id
            [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(
                stream_id
            )
            # The pose of the given camera at this timestamp is (world_camera = world_device @ device_camera):
            world_camera_pose = headset_pose3d.T_world_device @ extrinsics
            return [world_camera_pose, intrinsics]
    return None


def retrieve_hand_data(timestamp_ns: int) -> Optional[HandPose3dCollection]:
    """
    Retrieve the collection of Hand Pose at this timestamp (i.e. LEFT or RIGHT hand)
    Note: They are 3D pose in world, and does not say if they are visible for a given camera or not (stream_id)
    Visibility can either being determined by using camera visibility (are vertices visible), or using the 2d hands bounding box
    """
    hand_poses_with_dt = None
    hand_poses_with_dt = (
        hot3d_data_provider.umetrack_hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
    )

    if hand_poses_with_dt is not None:
        return hand_poses_with_dt.pose3d_collection
    return None


if __name__ == "__main__":
    # ---------------------------
    # Paths & IO
    # ---------------------------
    IMG_OUT_DIR = "../data/egoman_dataset/hot3d-grab-images"
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    META_PKL_IN = "../data/egoman_dataset/hot3d-grab-meta.pkl"
    ANNO_PKL_OUT = "../data/egoman_dataset/hot3d-grab-anno-close.pkl"
    HOT3D_ROOT = "../data/egoman_dataset/hot3d_v20241129_dec18_v4"
    time_folder = "../data/egoman_dataset/hot3d_anno/timepoints"

    os.makedirs(os.path.dirname(ANNO_PKL_OUT), exist_ok=True)


    # extract timepoints from HOT3D
    for clip_name in tqdm(
        os.listdir(f"{HOT3D_ROOT}/aria/")
    ):
        if clip_name.endswith(".json"):
            continue
        csv_path = f"{HOT3D_ROOT}/aria/{clip_name}/dynamic_objects.csv"
        metadata_path = f"{HOT3D_ROOT}/aria/{clip_name}/metadata.json"

        # Perform analysis
        results = analyze_dynamic_objects(csv_path, metadata_path)

        # Save results to JSON
        output_path = f"{time_folder}/{clip_name}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if len(results["objects"]) == 0:
            continue
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # extract meta data from HOT3D
    all_data = {}
    for cur_file in tqdm(os.listdir(time_folder)):
        cur_file_path = os.path.join(time_folder, cur_file)
        with open(cur_file_path, "r") as f:
            cur_data = json.load(f)
        video_name = cur_file[:-5]
        all_end_time = {}
        for item_obj in cur_data["objects"]:
            cur_anno = cur_data["objects"][item_obj]
            for seg in cur_anno["segments"]:
                all_end_time[seg["end_time_s"]] = cur_anno["object_uid"]
        sorted_end_time = sorted([key for key in all_end_time.keys()])
        for item_obj in cur_data["objects"]:
            cur_anno = cur_data["objects"][item_obj]
            for seg in cur_anno["segments"]:
                last_idx = max(
                    0, bisect.bisect_right(sorted_end_time, seg["start_time_s"]) - 1
                )
                last_end_time = sorted_end_time[last_idx]
                last_obj_uid = all_end_time[last_end_time]
                if last_obj_uid != cur_anno["object_uid"]:
                    all_data.setdefault(video_name, {})
                    all_data[video_name].setdefault(cur_anno["object_uid"], [])
                    cur_data_item = {
                        "dataset": "hot3d-grab",
                        "video_name": video_name,
                        "object_name": item_obj,
                        "object_uid": cur_anno["object_uid"],
                        "start_time": seg["start_time_s"],
                        "end_time": seg["end_time_s"],
                        "last_end_time": last_end_time,
                        "last_obj_uid": last_obj_uid,
                    }
                    all_data[video_name][cur_anno["object_uid"]].append(cur_data_item)
    pickle.dump(all_data, open(META_PKL_IN, "wb"))

    # extract trajectory
    hot3d_all_annos = []
    raw_meta = pickle.load(open(META_PKL_IN, "rb"))

    object_library_path = f"{HOT3D_ROOT}/object_library/assets"

    # Cache for providers/calibration per clip
    provider_cache = {}

    for clip_name in tqdm(raw_meta):
        aria_root = (
            f"{HOT3D_ROOT}/aria/{clip_name}"
        )
        csv_path = f"{aria_root}/dynamic_objects.csv"
        metadata_path = f"{aria_root}/metadata.json"
        hand_path = f"{aria_root}/umetrack_hand_pose_trajectory.jsonl"
        vrs_path = f"{aria_root}/recording.vrs"
        mps_folder = f"{aria_root}/mps"
        cam_file = f"{aria_root}/camera_models.json"
        # Init providers once per clip
        if clip_name not in provider_cache:
            mps_paths = mps.MpsDataPathsProvider(mps_folder).get_data_paths()
            mps_provider = mps.MpsDataProvider(mps_paths)
            mapping_csv_path = os.path.join(aria_root, "timecode_devicetime_mapping.csv")
            assert os.path.exists(
                mapping_csv_path
            ), f"Missing mapping file at {mapping_csv_path}"
            tc2dev_mapping = load_timecode_device_mapping(mapping_csv_path)

            vrs_provider = data_provider.create_vrs_data_provider(vrs_path)

            # Choose the stream used for image/extrinsics/intrinsics
            if USE_STREAM_ID:
                from projectaria_tools.core.stream_id import StreamId

                target_stream = StreamId(STREAM_ID_STR)
                target_label = vrs_provider.get_label_from_stream_id(target_stream)
            else:
                target_stream = None
                target_label = IMAGE_STREAM_LABEL

            # Get device->sensor (camera) extrinsics for the chosen stream/label
            T_device_sensor = (
                vrs_provider.get_device_calibration()
                .get_transform_device_sensor(target_label)
                .to_matrix()
            )
            T_sensor_device = np.linalg.inv(T_device_sensor)

            # First device-time timestamp for the chosen stream (for convenience)
            first_device_time_ns = None
            if target_stream is not None:
                first_device_time_ns = vrs_provider.get_first_time_ns(
                    target_stream, TimeDomain.DEVICE_TIME
                )
            else:
                # find the first time on any stream with this label
                for sid in vrs_provider.get_all_streams():
                    if vrs_provider.get_label_from_stream_id(sid) == target_label:
                        first_device_time_ns = vrs_provider.get_first_time_ns(
                            sid, TimeDomain.DEVICE_TIME
                        )
                        break
            assert (
                first_device_time_ns is not None
            ), "Could not find first device timestamp for target stream"

            # Also keep the intrinsics if you plan to project to 2D (for overlay)
            intrinsics = None
            try:
                if target_stream is not None:
                    extr, intr = vrs_provider.get_camera_calibration(target_stream)
                else:
                    # find a stream id by label to get full calibration tuple
                    chosen_sid = None
                    for sid in vrs_provider.get_all_streams():
                        if vrs_provider.get_label_from_stream_id(sid) == target_label:
                            chosen_sid = sid
                            break
                    if chosen_sid is not None:
                        extr, intr = vrs_provider.get_camera_calibration(chosen_sid)
                    else:
                        extr, intr = None, None
                intrinsics = intr
            except Exception:
                intrinsics = None  # overlay projection fallback will be pinhole if desired


            # Init the object library
            #
            object_library = load_object_library(
                object_library_folderpath=object_library_path
            )

            mano_hand_model = None

            #
            # Initialize hot3d data provider
            #
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=aria_root,
                object_library=object_library,
                mano_hand_model=mano_hand_model,
            )
            device_pose_provider = hot3d_data_provider.device_pose_data_provider
            device_data_provider = hot3d_data_provider.device_data_provider

            provider_cache[clip_name] = dict(
                mps=mps_provider,
                vrs=vrs_provider,
                target_stream=target_stream,
                target_label=target_label,
                T_device_sensor=T_device_sensor,
                T_sensor_device=T_sensor_device,
                first_device_time_ns=first_device_time_ns,
                intrinsics=intrinsics,
                tc2dev_mapping=tc2dev_mapping,
                device_pose_provider=device_pose_provider,
                device_data_provider=device_data_provider,
            )

        cached = provider_cache[clip_name]
        mps_provider = cached["mps"]
        vrs_provider = cached["vrs"]
        target_stream = cached["target_stream"]
        target_label = cached["target_label"]
        T_device_sensor = cached["T_device_sensor"]
        T_sensor_device = cached["T_sensor_device"]
        first_device_time_ns = cached["first_device_time_ns"]
        intrinsics = cached["intrinsics"]
        device_pose_provider = cached["device_pose_provider"]
        device_data_provider = cached["device_data_provider"]

        # Load metadata and build name→uid map
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        object_mapping = {
            name: uid
            for name, uid in zip(metadata["object_names"], metadata["object_uids"])
        }

        # Load object tracks (world frame)
        df = pd.read_csv(csv_path)
        df["timestamp[ns]"] = df["timestamp[ns]"].astype(np.int64)
        start_time_ns = df["timestamp[ns]"].min()

        # Load hand tracks (world frame)
        all_hand_traj = {"left": {}, "right": {}}
        with open(hand_path, "r") as f:
            for line in f:
                d = json.loads(line)
                ts_ns = int(d["timestamp_ns"])  # assumed TIME_CODE in source
                # Convert TIME_CODE to device-time seconds offset from first_device_time_ns
                # If your JSONL timestamps are already DEVICE_TIME, drop the conversion line below.
                t_s = (ts_ns - start_time_ns) / 1e9

                for hand_id, info in d["hand_poses"].items():
                    hand_key = "left" if str(hand_id) == "0" else "right"
                    wrist = info["wrist_xform"]
                    # q_wxyz -> xyzw
                    q_xyzw = [
                        wrist["q_wxyz"][1],
                        wrist["q_wxyz"][2],
                        wrist["q_wxyz"][3],
                        wrist["q_wxyz"][0],
                    ]
                    all_hand_traj[hand_key][t_s] = dict(
                        loc=wrist["t_xyz"],
                        rot=quat_normalize_xyzw(q_xyzw),
                        confidence=info["hand_confidence"],
                    )

        # Sort dicts by time
        for k in ("left", "right"):
            all_hand_traj[k] = dict(sorted(all_hand_traj[k].items()))

        # Iterate objects listed in meta
        for obj_uid in raw_meta[clip_name]:
            # Build object traj dict for this uid (world frame)
            all_obj_traj = {}
            for object_name, object_uid2 in object_mapping.items():
                if object_uid2 != obj_uid:
                    continue
                sel = (
                    df[df["object_uid"] == int(object_uid2)]
                    .copy()
                    .sort_values("timestamp[ns]")
                )
                pos_cols = ["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]
                for _, row in sel.iterrows():
                    pos = row[pos_cols].values.astype(np.float64)
                    # q_wo_* is stored wxyz -> convert to xyzw
                    rot = [row["q_wo_x"], row["q_wo_y"], row["q_wo_z"], row["q_wo_w"]]
                    ts_tc_ns = int(row["timestamp[ns]"])  # likely TIME_CODE
                    t_s = (ts_tc_ns - start_time_ns) / 1e9
                    all_obj_traj.setdefault(object_uid2, {})[t_s] = (
                        pos,
                        quat_normalize_xyzw(rot),
                    )
            if len(all_obj_traj) == 0:
                continue

            # Process each annotation block
            for cur_anno in raw_meta[clip_name][obj_uid]:

                def transform_seq_world_to_cam_single_frame(
                    seq_world: np.ndarray,
                    T_world_camera_se3,  # SE3 from retrieve_device_pose()
                ) -> np.ndarray:
                    """
                    Convert a sequence of poses from WORLD frame to the single CAMERA frame at question_sec.

                    Args:
                        seq_world: (T,7) with [x,y,z, qx,qy,qz,qw] in WORLD coordinates.
                        T_world_camera_se3: SE3 that maps CAMERA -> WORLD (i.e., T_world_camera).
                                            From the tutorial code: world_camera_pose = T_world_device @ T_device_camera

                    Returns:
                        seq_cam: (T,7) with [x,y,z, qx,qy,qz,qw] in CAMERA coordinates (at question frame).
                    """
                    # Invert to get CAMERA -> WORLD inverse, i.e., WORLD -> CAMERA
                    T_camera_world_se3 = T_world_camera_se3.inverse()

                    # Pull numeric rotation/translation from the SE3 inverse
                    # (Sophus SE3 gives .matrix() as 4x4; .so3().matrix() for 3x3; .translation() for 3,)
                    T_cw = T_camera_world_se3.to_matrix()
                    R_cw = T_cw[:3, :3]
                    t_cw = T_cw[:3, 3]

                    out = np.zeros_like(seq_world, dtype=np.float64)
                    for i in range(seq_world.shape[0]):
                        # position
                        p_w = seq_world[i, :3]
                        p_c = R_cw @ p_w + t_cw

                        # orientation
                        q_w = seq_world[i, 3:7]  # xyzw in WORLD
                        R_w = quat_xyzw_to_R(q_w)  # WORLD rotation matrix
                        R_c = R_cw @ R_w  # rotate into CAMERA frame
                        q_c = R_to_quat_xyzw(R_c)  # back to xyzw

                        out[i, :3] = p_c
                        out[i, 3:7] = q_c
                    return out, T_world_camera_se3

                start_sec = float(cur_anno["start_time"])
                end_sec = min(float(cur_anno["end_time"]), start_sec + 5.0)

                _, l0 = find_closest_time(all_hand_traj["left"], start_sec)
                _, r0 = find_closest_time(all_hand_traj["right"], start_sec)
                _, (op0, _) = find_closest_time(all_obj_traj[obj_uid], start_sec)
                left_dist = np.linalg.norm(np.array(l0["loc"]) - np.array(op0))
                right_dist = np.linalg.norm(np.array(r0["loc"]) - np.array(op0))
                manipulating_hand = "left" if left_dist < right_dist else "right"

                # find the farthest past timepoint that the target object is visible
                gap_time_int = [2, 1.5, 1, 0.5]
                for gap_time in gap_time_int:
                    question_sec = max(
                        start_sec - gap_time, float(cur_anno["last_end_time"]) + 0.25
                    )
                    _, (op00, _) = find_closest_time(all_obj_traj[obj_uid], question_sec)
                    if manipulating_hand == "left":
                        _, l00 = find_closest_time(all_hand_traj["left"], question_sec)
                        _dist0 = np.linalg.norm(np.array(l00["loc"]) - np.array(op00))
                    else:
                        _, r00 = find_closest_time(all_hand_traj["right"], question_sec)
                        _dist0 = np.linalg.norm(np.array(r00["loc"]) - np.array(op00))

                    if _dist0 < 1:
                        hand_timecode_ns = int(round(start_time_ns + question_sec * 1e9))
                        device_pose = retrieve_device_pose(
                            hand_timecode_ns,
                            image_streamid,
                            device_pose_provider,
                            device_data_provider,
                        )
                        device_pose_extrinsic = device_pose[0]
                        device_pose_intrinsic = device_pose[1]

                        op00 = np.concatenate([op00, [0, 0, 0, 1]]).astype(np.float64)
                        op00_cam, _ = transform_seq_world_to_cam_single_frame(
                            op00[None, ...], device_pose_extrinsic
                        )
                        if device_pose_intrinsic.project(op00_cam[0, :3]) is not None:
                            break

                hand_timecode_ns = int(round(start_time_ns + question_sec * 1e9))
                device_pose = retrieve_device_pose(
                    hand_timecode_ns,
                    image_streamid,
                    device_pose_provider,
                    device_data_provider,
                )
                device_pose_extrinsic = device_pose[0]
                device_pose_intrinsic = device_pose[1]

                op00 = np.concatenate([op00, [0, 0, 0, 1]]).astype(np.float64)
                op00_cam, _ = transform_seq_world_to_cam_single_frame(
                    op00[None, ...], device_pose_extrinsic
                )
                if device_pose_intrinsic.project(op00_cam[0, :3]) is None:
                    continue

                past_sec = max(0.0, question_sec - 0.5)

                # 10 fps sampling
                times = np.arange(past_sec, end_sec + 1e-6, 0.1)

                # Build world-frame sequences at 10 fps
                left_world, right_world, obj_world = [], [], []
                for t in times:
                    _, l = find_closest_time(all_hand_traj["left"], t)
                    _, r = find_closest_time(all_hand_traj["right"], t)
                    _, (op, oq) = find_closest_time(all_obj_traj[obj_uid], t)
                    left_world.append(np.array(l["loc"] + list(l["rot"]), dtype=np.float64))
                    right_world.append(
                        np.array(r["loc"] + list(r["rot"]), dtype=np.float64)
                    )
                    obj_world.append(np.concatenate([op, oq]).astype(np.float64))
                left_world = np.stack(left_world)  # (T,7)
                right_world = np.stack(right_world)  # (T,7)
                obj_world = np.stack(obj_world)  # (T,7)

                # Determine manipulating hand at start

                question_timecode_ns = int(round(first_device_time_ns + question_sec * 1e9))
                hand_timecode_ns = int(round(start_time_ns + question_sec * 1e9))
                device_pose = retrieve_device_pose(
                    hand_timecode_ns,
                    image_streamid,
                    device_pose_provider,
                    device_data_provider,
                )
                ori_hand_data = retrieve_hand_data(hand_timecode_ns)
                device_pose_extrinsic = device_pose[0]
                device_pose_intrinsic = device_pose[1]

                left_cam, T_sensor_world_question = transform_seq_world_to_cam_single_frame(
                    left_world, device_pose_extrinsic
                )
                right_cam, _ = transform_seq_world_to_cam_single_frame(
                    right_world, device_pose_extrinsic
                )
                obj_cam, _ = transform_seq_world_to_cam_single_frame(
                    obj_world, device_pose_extrinsic
                )

                # ---------------------------
                # Save annotation record
                # ---------------------------
                cur_rec = cur_anno.copy()
                cur_rec["times"] = times.tolist()
                cur_rec["left_hand_traj"] = left_cam
                cur_rec["right_hand_traj"] = right_cam
                cur_rec["object_traj"] = obj_cam
                cur_rec["manipulating_hand"] = manipulating_hand
                cur_rec["camera_T_world_at_question"] = T_sensor_world_question
                cur_rec["camera_label"] = target_label

                first_timecode_ns_for_this_stream = vrs_provider.get_first_time_ns(
                    stream_id, TimeDomain.TIME_CODE
                )
                # Fetch the exact image at question time from the same stream
                question_timecode_ns = (
                    int(round(question_sec * 1e9)) + first_timecode_ns_for_this_stream
                )  # or however you derive it

                # Fetch the closest image by TIME_CODE
                img_data = vrs_provider.get_image_data_by_time_ns(
                    stream_id,
                    question_timecode_ns,
                    TimeDomain.TIME_CODE,
                    TimeQueryOptions.CLOSEST,
                )
                arr = img_data[0].to_numpy_array()  # unpack tuple
                if arr is not None:
                    arr = arr.astype("uint8")
                    img = Image.fromarray(arr.astype(np.uint8))

                # Also copy your pre-extracted reference image, if you want to keep that convention
                frame_idx = int(round(question_sec * 4)) * 7
                img_name = f"{clip_name}_{obj_uid}_{round(question_sec, 3)}.jpg"
                dst_img = os.path.join(IMG_OUT_DIR, img_name)
                cur_rec["image"] = img_name

                # save image rot clockwise 90
                img = img.rotate(270)
                img.save(dst_img)

                # Save annotation record
                hot3d_all_annos.append(cur_rec)


    # Write once at the end
    with open(ANNO_PKL_OUT, "wb") as f:
        pickle.dump(hot3d_all_annos, f)
    print(f"Saved {len(hot3d_all_annos)} records to {ANNO_PKL_OUT}")
