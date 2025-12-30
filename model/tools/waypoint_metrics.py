# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os
import pickle
from pathlib import Path

import cv2

import numpy as np
from tqdm import tqdm


# -------- rotation: CCW 90° around camera Z (optical) axis --------
ROTATE_CCW_90 = True
_RZ_CCW90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot_ccw90_z(vec):
    """vec: (..., 3) -> (..., 3), apply +90° CCW around Z."""
    v = np.asarray(vec, dtype=float)
    if v.ndim == 1:
        return (_RZ_CCW90 @ v.reshape(3, 1)).reshape(3)
    else:
        return v @ _RZ_CCW90.T


def shift_toward(p: np.ndarray, target: np.ndarray, r: float) -> np.ndarray:
    """Shift p toward target by at most r (in meters)."""
    v = target - p
    d = float(np.linalg.norm(v))
    if d <= r or d == 0.0:
        return target.copy()
    return p + (r / d) * v


def nearest_point_on_traj(
    p: np.ndarray, traj: np.ndarray
) -> tuple[np.ndarray, float, int]:
    """
    Return (q, dist, idx) where q is the nearest point on traj to p.
    Uses per-sample NN among trajectory samples (no segment projection).
    """
    diffs = traj - p[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    i = int(np.argmin(dists))
    return traj[i], float(dists[i]), i


def project_point_fisheye_aria_np(point_xyz, params15, image_size):
    point_xyz = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    if point_xyz[2] <= 0:
        return None  # behind camera

    f = float(params15[0])
    cu = float(params15[1])
    cv = float(params15[2])
    k = np.asarray(params15[3:9], dtype=np.float64)
    p0, p1 = float(params15[9]), float(params15[10])
    s0, s1, s2, s3 = [float(x) for x in params15[11:15]]

    x, y, z = point_xyz
    eps = 1e-12

    a = x / z
    b = y / z
    r = np.hypot(a, b)
    th = np.arctan(r)

    if r < eps:
        th_div_r_a, th_div_r_b = 1.0, 0.0
    else:
        th_div_r_a, th_div_r_b = a / r, b / r

    th_k = th
    for i in range(6):
        th_k += k[i] * (th ** (3 + 2 * i))

    xr = th_k * th_div_r_a
    yr = th_k * th_div_r_b

    rd_sq = xr * xr + yr * yr
    rd_4 = rd_sq * rd_sq

    tu = (2.0 * xr * xr + rd_sq) * p0 + 2.0 * xr * yr * p1
    tv = (2.0 * yr * yr + rd_sq) * p1 + 2.0 * xr * yr * p0

    tp_u = s0 * rd_sq + s1 * rd_4
    tp_v = s2 * rd_sq + s3 * rd_4

    u = f * (xr + tu + tp_u) + cu
    v = f * (yr + tv + tp_v) + cv

    W, H = int(image_size[0]), int(image_size[1])
    if not (0 <= u < W and 0 <= v < H):
        return None
    return np.array([u, v], dtype=np.float32)


def pick_two_waypoints(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray | None:
    """
    If pred_pts has:
      - 0: None
      - 1: duplicate it -> (2,3)
      - 2: return as-is
      - >2: interpret as candidate pairs (0,1), (2,3), ... and
            choose the pair minimizing ||c-gt_c|| + ||f-gt_f||.
    """
    if pred_pts is None or pred_pts.size == 0:
        return None
    n = pred_pts.shape[0]
    if n == 1:
        return np.vstack([pred_pts[0], pred_pts[0]])
    if n == 2:
        return pred_pts

    # n >= 3: choose best pair among (0,1), (2,3), (4,5), ...
    best_score = None
    best_pair = None
    # step by 2; if odd length, last one is ignored (no final partner)
    for i in range(0, n - 1, 2):
        c = pred_pts[i]
        f = pred_pts[i + 1]
        score = np.linalg.norm(c - gt_pts[0]) + np.linalg.norm(f - gt_pts[1])
        if (best_score is None) or (score < best_score):
            best_score = score
            best_pair = np.vstack([c, f])

    # fallback if something weird happens
    if best_pair is None:
        best_pair = np.vstack([pred_pts[0], pred_pts[-1]])
    return best_pair


def draw_3d_point(
    img, pos_xyz, params15, image_size, color=(0, 0, 255), rotate_pixels_90cw=False
):
    pix = project_point_fisheye_aria_np(pos_xyz, params15, image_size)
    if pix is None:
        return img, False
    u, v = float(pix[0]), float(pix[1])
    if rotate_pixels_90cw:
        W, H = int(image_size[0]), int(image_size[1])
        x = int(round(W - 1 - v))
        y = int(round(u))
    else:
        x = int(round(u))
        y = int(round(v))
    cv2.circle(img, (x, y), 8, color, -1)
    cv2.circle(img, (x, y), 8, (255, 255, 255), 2)
    return img, True


def mname(p):
    return os.path.splitext(os.path.basename(p))[0].split("_")[0]


def as_np3(x):
    if x is None:
        return None
    a = np.asarray(x, float)
    if a.ndim == 1 and a.size == 3:
        a = a.reshape(1, 3)
    if a.ndim != 2 or a.shape[1] != 3:
        return None
    return a


def two_pts(pts):
    if pts is None or pts.size == 0:
        return None
    if pts.shape[0] == 1:
        return np.vstack([pts[0], pts[0]])
    return np.vstack([pts[0], pts[-1]])


def nn_to_traj(pts2, traj):
    if traj is None or traj.size == 0:
        return np.nan, np.nan
    d = np.linalg.norm(traj[None, :, :] - pts2[:, None, :], axis=2)
    m = d.min(axis=1)
    return float(m[0]), float(m[1])


# -------- load annotations --------
annotations = pickle.load(
    open("../data/egoman_dataset/egoman-test-final.pkl", "rb")
) + pickle.load(open("../data/egoman_dataset/hot3d-grab-anno-clean.pkl", "rb"))
annotations = {anno["image"]: anno for anno in annotations}

# -------- load EgoMAN preds --------
egoman_pred = pickle.load(open("../output/EgoMAN-7B-egomanbench.pkl", "rb"))

gt_all_dict = {}
egoman_wp_dict = {}

for pred in tqdm(egoman_pred):
    img_key = pred["image"]
    gt_anno = annotations[img_key]

    # decide left vs right hand from text
    is_left = ("left hand" in pred["value"][2].lower()) and (
        "right hand" not in pred["value"][2].lower()
    )

    if is_left:
        gt_3d_traj = gt_anno["value"][-2][5:, :3]
        try:
            gt_3d = [gt_anno["value"][0][1][1:4], gt_anno["value"][0][2][1:4]]
        except Exception:
            contact_time = round((gt_anno["start_time"] - gt_anno["times"][0]) * 10) - 5
            if contact_time >= len(gt_3d_traj):
                continue
            gt_3d = [gt_3d_traj[contact_time], gt_3d_traj[-1]]
        pred_3d = [np.array(pred["pred_contact"][1:4]), np.array(pred["pred_end"][1:4])]
    else:
        gt_3d_traj = gt_anno["value"][-2][5:, 3:]
        try:
            gt_3d = [gt_anno["value"][0][1][4:7], gt_anno["value"][0][2][4:7]]
        except Exception:
            contact_time = round((gt_anno["start_time"] - gt_anno["times"][0]) * 10) - 5
            if contact_time >= len(gt_3d_traj):
                continue
            gt_3d = [gt_3d_traj[contact_time], gt_3d_traj[-1]]
        pred_3d = [np.array(pred["pred_contact"][4:7]), np.array(pred["pred_end"][4:7])]

    # ------- align by rotating GT and EgoMAN waypoints/trajectory CCW 90° before saving -------
    if ROTATE_CCW_90:
        # trajectory: (T,3)
        gt_3d_traj = rot_ccw90_z(gt_3d_traj)
        # waypoints: list of 2 vectors
        gt_3d = [rot_ccw90_z(gt_3d[0]), rot_ccw90_z(gt_3d[1])]
        pred_3d = [rot_ccw90_z(pred_3d[0]), rot_ccw90_z(pred_3d[1])]

    # save packed dicts
    gt_all_dict[img_key] = {
        "text": pred["value"][2],
        "3d": gt_3d,  # [contact, final] after rotation (if enabled)
        "traj": gt_3d_traj,  # (T,3) after rotation (if enabled)
    }
    egoman_wp_dict[img_key] = {
        "text": pred["value"][2],
        "3d": pred_3d,  # [contact, final] after rotation (if enabled)
    }


# -------- dump results --------
pickle.dump(gt_all_dict, open("../output/gt_EgoMAN-7B-egomanbench_wp_res.pkl", "wb"))
pickle.dump(
    egoman_wp_dict, open("../output/egoman_EgoMAN-7B-egomanbench_wp_res.pkl", "wb")
)

FILES = [
    "../output/gt_EgoMAN-7B-egomanbench_wp_res.pkl",
    "../output/egoman_EgoMAN-7B-egomanbench_wp_res.pkl",
]
NAME = {
    "egoman": "EgoMAN",
    "gt": "GT",
}

COLORS = {
    "GT": (0, 255, 0),
    "EgoMAN": (0, 0, 255),
}

cam_intrinsics = pickle.load(open("../data/egoman_dataset/egoman_cam_params.pkl", "rb"))
cam_intrinsics.update(
    pickle.load(open("../data/egoman_dataset/egoman_cam_params_hot3d.pkl", "rb"))
)

ROTATE_PIXELS_90CW = False
OUT_DIR = Path("../output/wp_proj_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="If set, load images and draw waypoints.",
    )
    ap.add_argument(
        "--shift",
        action="store_true",
        help="If set, allow fairness wrist shift for non-GT/non-EgoMAN.",
    )
    ap.add_argument(
        "--shift_radius",
        type=float,
        default=0.06,
        help="Max wrist shift radius in meters (default 0.06).",
    )
    args = ap.parse_args()

    # load all pickles
    bags = {}
    for p in FILES:
        with open(p, "rb") as f:
            bags[NAME[mname(p)]] = pickle.load(f)
    # strict intersection
    key_sets = {m: set(d.keys()) for m, d in bags.items()}
    common = sorted(set.intersection(*key_sets.values()))
    print(
        "[key counts]", {m: len(s) for m, s in key_sets.items()}, "common:", len(common)
    )

    methods = [m for m in bags if m != "GT"]

    split_tags = ["egoman_unseen", "hot3d_ood"]
    sums = {m: {tag: np.zeros(4) for tag in split_tags} for m in methods}
    cnts = {m: {tag: np.zeros(4, int) for tag in split_tags} for m in methods}

    count = 0
    for k in tqdm(common):
        # dataset routing + image path
        image_path = "../data/egomanbench_imgs/" + k
        video_id = k.split("_videostamp")[0]
        split_tag = "egoman_unseen"
        if video_id not in cam_intrinsics:
            parts = k.split("_")
            video_id = "_".join(parts[:2])
            split_tag = "hot3d_ood"

        if video_id not in cam_intrinsics:
            continue

        cur_aria_cam = cam_intrinsics[video_id]
        params15 = cur_aria_cam[0]
        if isinstance(params15[0], (list, tuple, np.ndarray)):
            params15 = params15[0]
        image_size = tuple(cur_aria_cam[1])

        # read image only if visualize
        if args.visualize:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            W, H = int(image_size[0]), int(image_size[1])
            if img.shape[1] != W or img.shape[0] != H:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        # GT fields (required)
        gt = bags["GT"][k]
        gt_pts = two_pts(as_np3(gt.get("3d")))
        gt_traj = as_np3(gt.get("traj"))
        if gt_pts is None or gt_pts.shape != (2, 3) or gt_traj is None:
            continue

        if args.visualize:
            draw_3d_point(
                img,
                gt_pts[0],
                params15,
                image_size,
                COLORS["GT"],
                rotate_pixels_90cw=ROTATE_PIXELS_90CW,
            )
            draw_3d_point(
                img,
                gt_pts[1],
                params15,
                image_size,
                COLORS["GT"],
                rotate_pixels_90cw=ROTATE_PIXELS_90CW,
            )
            cv2.putText(
                img,
                "GT",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLORS["GT"],
                2,
                cv2.LINE_AA,
            )
            y_text = 56
        else:
            y_text = 0  # unused

        for m in methods:
            pred = bags[m][k]
            pred_raw = as_np3(pred.get("3d"))
            pred_pts = pick_two_waypoints(pred_raw, gt_pts)
            if pred_pts is None or pred_pts.shape != (2, 3):
                A_c = A_f = B_c = B_f = np.nan
            else:
                # Raw distances
                A_c_raw = float(np.linalg.norm(pred_pts[0] - gt_pts[0]))
                A_f_raw = float(np.linalg.norm(pred_pts[1] - gt_pts[1]))
                B_c_raw, B_f_raw = nn_to_traj(pred_pts, gt_traj)

                if args.shift and (m not in ("GT", "EgoMAN")):
                    r = args.shift_radius

                    # ----- Endpoint metric (A): shift toward GT waypoints -----
                    predA0_shift = shift_toward(pred_pts[0], gt_pts[0], r)
                    predA1_shift = shift_toward(pred_pts[1], gt_pts[1], r)
                    A_c = float(np.linalg.norm(predA0_shift - gt_pts[0]))
                    A_f = float(np.linalg.norm(predA1_shift - gt_pts[1]))

                    # ----- Traj-NN metric (B): shift toward nearest traj points -----
                    q0, _, _ = nearest_point_on_traj(pred_pts[0], gt_traj)
                    q1, _, _ = nearest_point_on_traj(pred_pts[1], gt_traj)
                    predB0_shift = shift_toward(pred_pts[0], q0, r)
                    predB1_shift = shift_toward(pred_pts[1], q1, r)
                    # recompute NN distances after shifting
                    B_c = float(
                        np.min(np.linalg.norm(gt_traj - predB0_shift[None, :], axis=1))
                    )
                    B_f = float(
                        np.min(np.linalg.norm(gt_traj - predB1_shift[None, :], axis=1))
                    )
                else:
                    # No shift (or GT/EgoMAN): use raw
                    A_c, A_f, B_c, B_f = A_c_raw, A_f_raw, B_c_raw, B_f_raw

                # visualize points (actual predictions; we do not move them visually)
                if args.visualize:
                    color = COLORS.get(m, (255, 255, 255))
                    draw_3d_point(
                        img,
                        pred_pts[0],
                        params15,
                        image_size,
                        color,
                        rotate_pixels_90cw=ROTATE_PIXELS_90CW,
                    )
                    draw_3d_point(
                        img,
                        pred_pts[1],
                        params15,
                        image_size,
                        color,
                        rotate_pixels_90cw=ROTATE_PIXELS_90CW,
                    )
                    label = f"{m}{' (shift)' if (args.shift and m not in ('GT','EgoMAN')) else ''}"
                    cv2.putText(
                        img,
                        label,
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    y_text += 28
            # Contact: A_c; Traj: (B_f + B_c) / 2.0
            vals = np.array([A_c, A_f, B_c, (B_f + B_c) / 2.0], float)
            mask = ~np.isnan(vals)
            sums[m][split_tag][mask] += vals[mask]
            cnts[m][split_tag][mask] += 1

        if args.visualize:
            out_path = OUT_DIR / f"{k}"
            cv2.imwrite(str(out_path), img)
        count += 1
    print(count)
    print("\nPer-method waypoint metrics:")
    print("Method\tTest-Split\tContact\tTraj")
    for tag in split_tags:
        for m in sorted(methods):
            mean = np.where(
                cnts[m][tag] > 0, sums[m][tag] / np.maximum(cnts[m][tag], 1), np.nan
            )
            print(f"{m}\t{tag}\t{mean[0]:.4f}\t{mean[3]:.4f}")


if __name__ == "__main__":
    main()
