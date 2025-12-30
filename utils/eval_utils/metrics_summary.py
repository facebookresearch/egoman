# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import os
import pickle
import re
from collections import Counter

from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _image_set(samples: List[dict], image_key: str) -> Set[Any]:
    s = {d.get(image_key) for d in samples if image_key in d}
    s.discard(None)
    return s


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def filter_by_image_clusters(
    method2samples: Dict[str, List[dict]],
    image_key: str = "image",
    max_groups: int = 3,
    verbose: bool = True,
) -> Tuple[Dict[str, List[dict]], Dict[str, int], Dict[int, Set[Any]]]:
    """
    Cluster methods by their image sets (up to `max_groups` clusters). If a method has
    zero overlap with all current groups, start a new group (until `max_groups`);
    otherwise join the closest (highest Jaccard). Then filter each method to the
    selected images of its group.

    Group images selection:
      1) strict intersection across methods in the group
      2) if empty, fallback to images present in >= ceil(|group|/2) methods

    Returns:
        filtered_method2samples, method_to_group, group_to_images
    """
    # 1) Build per-method image sets
    method_imgs: Dict[str, Set[Any]] = {
        m: _image_set(samps, image_key) for m, samps in method2samples.items()
    }
    methods = list(method_imgs.keys())
    if not methods:
        return method2samples, {}, {}

    # 2) Greedy clustering
    group_reps: Dict[int, Set[Any]] = {}  # representative set per group
    method_to_group: Dict[str, int] = {}
    next_gid = 0

    for m in methods:
        imgs = method_imgs[m]
        if not group_reps:
            group_reps[next_gid] = set(imgs)
            method_to_group[m] = next_gid
            next_gid += 1
            continue

        # compute jaccard to existing groups
        overlaps = {gid: len(imgs & rep) for gid, rep in group_reps.items()}
        has_any_overlap = any(v > 0 for v in overlaps.values())
        if has_any_overlap:
            # join closest by Jaccard
            jaccs = {gid: _jaccard(imgs, rep) for gid, rep in group_reps.items()}
            best_gid = max(jaccs, key=jaccs.get)
            method_to_group[m] = best_gid
            # Optionally update representative as union to better reflect the group
            group_reps[best_gid] = group_reps[best_gid] | imgs
        else:
            # zero overlap with all existing groups
            if next_gid < max_groups:
                group_reps[next_gid] = set(imgs)
                method_to_group[m] = next_gid
                next_gid += 1
            else:
                # already at max_groups -> attach to closest by Jaccard (even if 0)
                jaccs = {gid: _jaccard(imgs, rep) for gid, rep in group_reps.items()}
                best_gid = max(jaccs, key=jaccs.get)
                method_to_group[m] = best_gid
                group_reps[best_gid] = group_reps[best_gid] | imgs

    # 3) For each group, choose images: strict intersection -> fallback majority
    group_members: Dict[int, List[str]] = {}
    for m, gid in method_to_group.items():
        group_members.setdefault(gid, []).append(m)

    group_to_images: Dict[int, Set[Any]] = {}
    for gid, members in group_members.items():
        sets = [method_imgs[m] for m in members]
        inter = set.intersection(*sets) if sets else set()
        if inter:
            selected = inter
            reason = f"intersection ({len(selected)})"
        else:
            # fallback: images that appear in >= ceil(|group|/2) methods
            k = math.ceil(len(members) / 2)
            ctr = Counter()
            for s in sets:
                for img in s:
                    ctr[img] += 1
            selected = {img for img, c in ctr.items() if c >= k}
            reason = f"majority ≥{k} methods ({len(selected)})"
        group_to_images[gid] = selected
        if verbose:
            print(
                f"[image-cluster] Group {gid}: members={members}, "
                f"strict_inter={len(inter)}, selected={len(selected)} via {reason}"
            )

    # 4) Filter samples per method by their group's selected images (dedup preserve order)
    filtered: Dict[str, List[dict]] = {}
    for m, samples in method2samples.items():
        gid = method_to_group[m]
        selected = group_to_images.get(gid, set())
        seen = set()
        kept = []
        for s in samples:
            img = s.get(image_key)
            if img in selected and img not in seen:
                kept.append(s)
                seen.add(img)
        filtered[m] = kept
        if verbose:
            total = len(
                [1 for s in samples if image_key in s and s.get(image_key) is not None]
            )
            print(f"[image-cluster] {m}: kept {len(kept)} / {total} (group {gid})")

    return filtered, method_to_group, group_to_images


def _infer_ratio(method: str) -> int:
    if method is None:
        return 100
    name = method.strip()
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 100


def _base_name(method: str) -> str:
    if method is None:
        return ""
    return re.sub(r"_(\d+)$", "", method.strip())


def add_ratio_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Method"] = df["Method"].astype(str).str.strip()
    df["DataRatio"] = df["Method"].apply(_infer_ratio).astype(int)
    df["Base"] = df["Method"].apply(_base_name)

    # Sort: by Base (A–Z), then DataRatio with custom order 100>50>25>10>5
    cat = pd.CategoricalDtype(RATIO_ORDER, ordered=True)
    df["DataRatioCat"] = df["DataRatio"].astype("Int64").astype(cat)
    df = df.sort_values(["Base", "DataRatioCat"], ascending=[True, True]).drop(
        columns=["DataRatioCat"]
    )

    # Optional: if you want just the method name order (including suffix), comment the sort above
    # and do: df = df.sort_values("Method", kind="stable")

    # Put DataRatio near Method
    cols = df.columns.tolist()
    cols.remove("DataRatio")
    cols.remove("Base")
    cols = ["Method", "Base", "DataRatio"] + cols[1:]  # keep existing metrics order
    return df[cols]


def scale_distance_metrics(df: pd.DataFrame, factor: float = 100.0) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    cols_to_scale = [
        "ADE",
        "FDE",
        "DTW",
    ]
    for c in cols_to_scale:
        if c in df.columns:
            df[c] = df[c].astype(float) * factor
    return df


def _normalize(q, eps=1e-12):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def max_deg_from_start(q_all_xyzw: np.ndarray) -> Tuple[float, int, np.ndarray]:
    q_all = _normalize(q_all_xyzw.astype(np.float64))
    q0 = q_all[0]
    dots = np.abs(np.sum(q_all * q0, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    angles_rad = 2.0 * np.arccos(dots)
    angles_deg = np.degrees(angles_rad)
    max_idx = int(np.argmax(angles_deg))
    return float(angles_deg[max_idx]), max_idx, angles_deg


def calculate_hand_distance(cur_data: dict) -> Tuple[float, float, float]:
    """
    Returns:
        ade (m), deg (deg), fde (m)
    Uses cur_data["value"] structure you described:
      value[-2] -> future traj (T, 6)  [x_l,y_l,z_l, x_r,y_r,z_r]
      value[-1] -> future quat (T, 8) [l_xyzw(4), r_xyzw(4)]
      start index is 5 (we measure displacement after frame 5)
    """
    fut_traj = np.asarray(cur_data["value"][-2])  # (T, 6)
    fut_quat = np.asarray(cur_data["value"][-1])  # (T, 8) as [l(4), r(4)]
    assert fut_traj.shape[0] >= 6 and fut_quat.shape[0] >= 6, "Need at least 6 frames."

    left_quat_all = fut_quat[5:, :4]
    right_quat_all = fut_quat[5:, 4:]
    left_max_deg, _, _ = max_deg_from_start(left_quat_all)
    right_max_deg, _, _ = max_deg_from_start(right_quat_all)

    left_positions = fut_traj[5:, :3]  # [T,3]
    right_positions = fut_traj[5:, 3:]  # [T,3]
    left_start = fut_traj[5, :3]
    right_start = fut_traj[5, 3:]

    left_dists = np.linalg.norm(left_positions - left_start, axis=1)
    right_dists = np.linalg.norm(right_positions - right_start, axis=1)

    left_ade = float(left_dists.mean())
    right_ade = float(right_dists.mean())

    left_fde = float(np.linalg.norm(left_positions[-1] - left_start))
    right_fde = float(np.linalg.norm(right_positions[-1] - right_start))

    return (
        (left_ade + right_ade) / 2.0,
        (left_max_deg + right_max_deg) / 2.0,
        (left_fde + right_fde) / 2.0,
    )


def categorize_sample(cur_data: dict) -> Tuple[str, str, float, float, float]:
    """
    Returns:
        dist_tag ('short'|'long'),
        rot_tag  ('short'|'long'),
        dist (m), deg (deg), fde (m)
    Thresholds follow your spec: dist>=0.15 -> long, deg>=60 -> long
    """
    dist, deg, fde = calculate_hand_distance(cur_data)
    dist_tag = "long" if dist >= 0.15 else "short"
    rot_tag = "long" if deg >= 60.0 else "short"
    return dist_tag, rot_tag, dist, deg, fde


# =========================
# ---- Metric helpers -----
# =========================

HandFlag = Literal["left", "right", "both"]
MetricName = Literal["ade", "fde", "dtw", "rot"]


def _metric_value_from_kres(
    kres: dict, metric: MetricName, hand: HandFlag
) -> Optional[float]:
    """Extract a scalar from one k-sample result according to metric+hand."""
    if metric == "rot":
        if hand == "left":
            v = kres.get("left_rot_error", None)
        elif hand == "right":
            v = kres.get("right_rot_error", None)
        else:
            # both: average available (fallback to overall if present)
            l = kres.get("left_rot_error", None)
            r = kres.get("right_rot_error", None)
            if l is None and r is None:
                v = kres.get("overall_rot_error", None)
            elif l is None:
                v = r
            elif r is None:
                v = l
            else:
                v = 0.5 * (l + r)
        return float(v) if v is not None else None

    # distance-like metrics
    left_key = f"left_hand_{metric}"
    right_key = f"right_hand_{metric}"
    if hand == "left":
        v = kres.get(left_key, None)
    elif hand == "right":
        v = kres.get(right_key, None)
    else:
        l = kres.get(left_key, None)
        r = kres.get(right_key, None)
        if l is None and r is None:
            print("bad")
            return None
        if l is None:
            v = r
        elif r is None:
            v = l
        else:
            v = 0.5 * (l + r)
    return float(v) if v is not None else None


def _stack_metric_arrays(
    k_list: List[dict],
    hand: HandFlag,
    k_cap: Optional[int],
    metrics: List[MetricName],
) -> Dict[MetricName, np.ndarray]:
    """Return {metric: (K,) array} with np.inf for missing."""
    items = k_list if k_cap is None else k_list[:k_cap]
    K = len(items)
    out: Dict[MetricName, np.ndarray] = {}
    for m in metrics:
        arr = np.empty(K, dtype=float)
        for i, kres in enumerate(items):
            v = _metric_value_from_kres(kres, m, hand)
            arr[i] = v if (v is not None and np.isfinite(v)) else np.inf
        out[m] = arr
    return out


def _choose_best_k_lexi(
    vals: Dict[MetricName, np.ndarray], order: List[MetricName]
) -> int:
    """Lexicographic min over metrics in 'order'."""
    # Start with all candidates
    idx = np.arange(vals[order[0]].shape[0])
    mask = np.ones_like(idx, dtype=bool)
    for m in order:
        arr = vals[m]
        # Among remaining, find minimal value
        current = arr[mask]
        if not np.any(np.isfinite(current)):
            # if all inf, skip this metric
            continue
        min_val = np.nanmin(current)
        # keep those equal to min_val
        keep_global = (arr == min_val) & mask
        if keep_global.sum() == 0:
            # robust fallback: keep the absolute min overall
            keep_global = arr == np.nanmin(arr)
        mask = keep_global
        # early exit if single left
        if mask.sum() == 1:
            break
    # choose first True index (stable)
    if mask.sum() == 0:
        return -1
    return int(np.nonzero(mask)[0][0])


def _choose_best_k_rank(vals: Dict[MetricName, np.ndarray]) -> int:
    """Rank-sum across metrics; lower is better."""
    K = next(iter(vals.values())).shape[0]
    total_rank = np.zeros(K, dtype=float)
    any_metric = False
    for m, arr in vals.items():
        # finite mask
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        any_metric = True
        # ranks among finite only
        finite_vals = arr[finite]
        # obtain ranks via argsort twice
        order = np.argsort(finite_vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, order.size + 1, dtype=float)
        # assign ranks back
        r_full = np.full(K, np.max(ranks) + 1.0, dtype=float)  # penalize non-finite
        r_full[finite] = ranks
        total_rank += r_full
    if not any_metric:
        return -1
    return int(np.argmin(total_rank))


def summarize_one_sample(
    traj_metrics: dict,
    metric: MetricName = "ade",
    hand: HandFlag = "both",
    k_cap: Optional[int] = None,
    *,
    mode: Literal["lexi", "rank"] = "lexi",
    lexi_order: List[MetricName] = ("ade", "fde", "dtw", "rot"),
) -> Tuple[float, float, int]:
    """
    Fast version:
      - mean_val: mean of requested metric over top-K
      - best_idx: chosen by combined distance (ade + rot) to find closest trajectory
      - best_val: requested metric at best_idx
    """
    k_list = traj_metrics.get("k_samples", []) or []
    if not k_list:
        return 0.0, 0.0, -1
    items = k_list if k_cap is None else k_list[:k_cap]
    if not items:
        return 0.0, 0.0, -1

    # Stack arrays once - use only the top K samples (items)
    metrics_all = [
        "dtw",
        "rot",
        "ade",
        "fde",
    ]
    vals = _stack_metric_arrays(items, hand, None, metrics_all)

    # mean of requested metric over K (ignore inf)
    req = vals[metric]
    mean_val = (
        float(np.mean(req[np.isfinite(req)])) if np.any(np.isfinite(req)) else 0.0
    )

    # Choose best trajectory by combining positional distance and rotation distance
    # Normalize both metrics to similar scales and sum them to find closest trajectory
    ade_vals = vals["ade"]
    rot_vals = vals["rot"]

    # Create combined distance metric (normalize rotation to meters, assuming 1 degree ≈ 0.001m)
    combined_dist = ade_vals + (rot_vals * 0.001)

    # Find the index with minimum combined distance
    finite_mask = np.isfinite(combined_dist)
    if np.any(finite_mask):
        best_idx = int(np.argmin(np.where(finite_mask, combined_dist, np.inf)))
    else:
        best_idx = -1

    if best_idx < 0 or best_idx >= req.shape[0]:
        return mean_val, 0.0, -1

    best_val = float(req[best_idx]) if np.isfinite(req[best_idx]) else 0.0
    return mean_val, best_val, best_idx


# =========================
# ---- Table building -----
# =========================


def summarize_file(
    samples: List[dict],
    hand: HandFlag = "both",
    k_cap: Optional[int] = None,
    category_flag: Optional[Literal["distance", "rot"]] = None,
    category_value: Optional[Literal["short", "long"]] = None,
) -> Dict[str, float]:
    """
    Build 4 best metrics:
        ADE, FDE, DTW, ROT(deg)
    Filters by category if requested.
    Returns dict + 'N' (#samples used)
    """
    rows = []
    for s in samples:
        # category filter
        if category_flag is not None and category_value is not None:
            try:
                dist_tag, rot_tag, *_ = categorize_sample(s)
            except Exception:
                continue
            if category_flag == "distance" and dist_tag != category_value:
                continue
            if category_flag == "rot" and rot_tag != category_value:
                continue

        tm = s.get("trajectory_metrics")
        if not tm:
            continue

        _, ade_b, _ = summarize_one_sample(tm, "ade", hand, k_cap)
        _, fde_b, _ = summarize_one_sample(tm, "fde", hand, k_cap)
        _, dtw_b, _ = summarize_one_sample(tm, "dtw", hand, k_cap)
        _, rot_b, _ = summarize_one_sample(tm, "rot", hand, k_cap)

        rows.append(
            {
                "ADE": ade_b,
                "FDE": fde_b,
                "DTW": dtw_b,
                "ROT": rot_b,
            }
        )

    if not rows:
        return {
            k: 0.0
            for k in [
                "ADE",
                "FDE",
                "DTW",
                "ROT",
            ]
        } | {"N": 0}

    # aggregate across samples
    out = {}
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else 0.0
    out["N"] = len(rows)
    return out


def build_table(
    res_path: str,
    hand: HandFlag = "both",
    k_cap: Optional[int] = None,
    category_flag: Optional[Literal["distance", "rot"]] = None,
    category_value: Optional[Literal["short", "long"]] = None,
    benchmark_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize a single pkl file's metrics based on different parameters.

    Args:
        root_dir: Directory containing the pkl file
        hand: Which hand to evaluate ("left", "right", or "both")
        k_cap: Maximum number of k-samples to consider
        category_flag: Filter by category ("distance" or "rot")
        category_value: Category value to filter ("short" or "long")
        benchmark_name: Dataset name to filter on (e.g., "hot3d" or "egoman")
                        If "egoman_all", creates separate rows for each benchmark split
    """
    # Load samples from pkl file
    all_res = pickle.load(open(res_path, "rb"))
    m2s = {"EgoMAN": all_res}

    if not m2s:
        return pd.DataFrame(
            columns=[
                "Method",
                "Benchmark",
                "N",
                "ADE",
                "FDE",
                "DTW",
                "ROT",
            ]
        )

    # Get all samples (should be just one method)
    all_samples = []
    method_name = None
    for method, samples in m2s.items():
        all_samples.extend(samples)
        if method_name is None:
            method_name = method

    # Handle egoman_all case: create separate rows for each benchmark split
    if benchmark_name is not None and benchmark_name.lower() == "egoman_all":
        # Get unique benchmark splits
        benchmark_splits = set()
        for s in all_samples:
            dataset = s.get("benchmark_split", "").lower()
            if dataset:
                benchmark_splits.add(dataset)
        # Create rows for each benchmark split
        rows = []
        for split in sorted(benchmark_splits):
            filtered_samples = [
                s for s in all_samples if s.get("benchmark_split", "").lower() == split
            ]

            summary = summarize_file(
                filtered_samples,
                hand=hand,
                k_cap=k_cap,
                category_flag=category_flag,
                category_value=category_value,
            )

            if summary["N"] > 0:
                row = {
                    "Method": method_name or "unknown",
                    "Benchmark": split,
                    **summary,
                }
                rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "Method",
                    "Benchmark",
                    "N",
                    "ADE",
                    "FDE",
                    "DTW",
                    "ROT",
                ]
            )

        df = pd.DataFrame(rows)
    else:
        # Original single-row behavior
        # Filter by benchmark_name (dataset) if specified
        if benchmark_name is not None:
            filtered_samples = []
            for s in all_samples:
                dataset = s.get("benchmark_split", "").lower()
                if benchmark_name.lower() in dataset:
                    filtered_samples.append(s)
            all_samples = filtered_samples

        # Summarize the filtered samples
        summary = summarize_file(
            all_samples,
            hand=hand,
            k_cap=k_cap,
            category_flag=category_flag,
            category_value=category_value,
        )

        if summary["N"] == 0:
            return pd.DataFrame(
                columns=[
                    "Method",
                    "Benchmark",
                    "N",
                    "ADE",
                    "FDE",
                    "DTW",
                    "ROT",
                ]
            )

        # Create DataFrame with single row
        row = {
            "Method": method_name or "unknown",
            "Benchmark": benchmark_name or "all",
            **summary,
        }
        df = pd.DataFrame([row])

    # Select and order columns
    cols = [
        "Method",
        "Benchmark",
        "N",
        "ADE",
        "FDE",
        "DTW",
        "ROT",
    ]
    df = df[cols]

    # Convert metric columns to float
    for c in cols:
        if c not in ("Method", "Benchmark", "N"):
            df[c] = df[c].astype(float)

    return df


# =========================
# --------- CLI ----------
# =========================

if __name__ == "__main__":
    res_path = "../output/EgoMAN-7B-egomanbench_processed.pkl"

    # ---- knobs you'll tweak most often ----
    HAND: HandFlag = "both"  # "left" | "right" | "both"
    CATEGORY_FLAG = None  # None | "distance" | "rot"
    CATEGORY_VALUE = None  # None | "short" | "long"
    DATA_FLAG = "egoman_all"  # "egoman_all" | "hot3d_ood" | "egoman_unseen"
    # Examples:
    # CATEGORY_FLAG, CATEGORY_VALUE = "distance", "long"
    # CATEGORY_FLAG, CATEGORY_VALUE = "rot", "short"

    # Create subtables for different K_CAP values
    k_cap_values = [1, 5, 10]
    all_dfs = []

    for k_cap in k_cap_values:
        df = build_table(
            res_path,
            hand=HAND,
            k_cap=k_cap,
            category_flag=CATEGORY_FLAG,
            category_value=CATEGORY_VALUE,
            benchmark_name=DATA_FLAG,
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
        out_path = f"../output/metrics_{DATA_FLAG}.csv"
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
