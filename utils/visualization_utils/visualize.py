# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Visualize predicted hand trajectories from inference results.
Loads pkl from inference_new_stage2_od_6dof_fast_robohack.py,
visualizes K predicted hand trajectories, and saves to K images.
"""

import functools
import json
import math
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.transforms as pt
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ----------------------------
# Rotation Transformer
# ----------------------------
class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        assert from_rep != to_rep
        assert from_rep in self.valid_reps and to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        fwd, inv = [], []
        if from_rep != "matrix":
            funcs = [
                getattr(pt, f"{from_rep}_to_matrix"),
                getattr(pt, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(fn, convention=from_convention) for fn in funcs
                ]
            fwd.append(funcs[0])
            inv.append(funcs[1])
        if to_rep != "matrix":
            funcs = [
                getattr(pt, f"matrix_to_{to_rep}"),
                getattr(pt, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [
                    functools.partial(fn, convention=to_convention) for fn in funcs
                ]
            fwd.append(funcs[0])
            inv.append(funcs[1])
        self.forward_funcs = fwd
        self.inverse_funcs = inv[::-1]

    @staticmethod
    def _apply_funcs(x, funcs):
        t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        for fn in funcs:
            t = fn(t)
        return t.numpy() if isinstance(x, np.ndarray) else t

    def forward(self, x):
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x):
        return self._apply_funcs(x, self.inverse_funcs)


tf = RotationTransformer(from_rep="rotation_6d", to_rep="quaternion")


# ----------------------------
# Camera projection
# ----------------------------
def project_point_fisheye_aria_np(point_xyz, params15, image_size=(1408, 1408)):
    """
    point_xyz: (3,) in camera coords (meters). +Z forward.
    params15: list/np.array of length 15:
        [f, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3]
      (single focal length; use fu=fv=f)
    image_size: (W, H)
    Returns (u, v) or None if behind camera or out of bounds.
    """
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    if z <= 0.0:
        return None

    p = np.asarray(params15, dtype=np.float64).reshape(-1)
    assert p.size == 15, "Expect 15 projection parameters (Fisheye624 single-f)."

    f = p[0]
    cu = p[1]
    cv = p[2]
    k = p[3:9]  # k0..k5
    p0 = p[9]
    p1 = p[10]
    s0, s1, s2, s3 = p[11], p[12], p[13], p[14]

    # Normalize
    a = x / z
    b = y / z
    r = math.hypot(a, b)
    eps = 1e-9

    # fisheye angle
    th = math.atan(r)
    if r < eps:
        dirx, diry = 1.0, 0.0  # any unit direction is fine at center
    else:
        dirx, diry = a / r, b / r

    # Kannala–Brandt radial
    th_k = th
    th_pow = th**3
    for i in range(6):  # k0..k5
        th_k += k[i] * th_pow
        th_pow *= th * th  # increment power by 2

    xr = th_k * dirx
    yr = th_k * diry
    rd2 = xr * xr + yr * yr
    rd4 = rd2 * rd2

    # Tangential
    tu = (2.0 * xr * xr + rd2) * p0 + 2.0 * xr * yr * p1
    tv = (2.0 * yr * yr + rd2) * p1 + 2.0 * xr * yr * p0

    # Thin-prism
    tp_u = s0 * rd2 + s1 * rd4
    tp_v = s2 * rd2 + s3 * rd4

    uvd_u = xr + tu + tp_u
    uvd_v = yr + tv + tp_v

    u = f * uvd_u + cu
    v = f * uvd_v + cv

    W, H = image_size
    if not (0 <= u < W and 0 <= v < H):
        return None
    return (u, v)


def rotate_90_clockwise(u, v, width, height):
    """Rotate pixel coordinates 90 degrees clockwise."""
    v_new = height - 1 - v
    return v_new, u


# ----------------------------
# Drawing helpers
# ----------------------------
BLUE = (255, 0, 0)  # BGR for left hand
RED = (0, 0, 255)  # BGR for right hand
YELLOW = (0, 255, 255)  # BGR for contact point
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GREEN_BGR = (40, 140, 40)  # rotation arrows

FUTURE_LINE_WIDTH = 28
DEFAULT_AXIS_LEN_M = 0.02
ROTATE_PIXELS_90CW = True


def fade_rgb_background(img_bgr, alpha_rgb=0.45):
    """Fade RGB background to white for overlay."""
    white = np.full_like(img_bgr, 255)
    return cv2.addWeighted(white, 1.0 - alpha_rgb, img_bgr, alpha_rgb, 0.0)


def overlay_traj_on_rgb(
    img_bgr,
    pred_seq_Tx18,
    cam_params,
    image_size,
    rotate90,
    contact_t=None,
    draw_contact=False,
    axis_len_m=DEFAULT_AXIS_LEN_M,
):
    """
    Overlay predicted hand trajectory on RGB image.

    Args:
        img_bgr: Base RGB image
        pred_seq_Tx18: (T, 18) predicted trajectory - [L_xyz(0:3), L_r6d(3:9), R_xyz(9:12), R_r6d(12:18)]
        cam_params: Camera projection parameters
        image_size: (W, H)
        rotate90: Whether to rotate pixels 90 degrees clockwise
        contact_t: Contact time step index (optional, for yellow circle marker)
        axis_len_m: Length of rotation axes in meters

    Returns:
        Annotated image
    """
    canvas = fade_rgb_background(img_bgr, 0.45)
    H, W = canvas.shape[:2]
    T = pred_seq_Tx18.shape[0]

    def split_row(row):
        # [L_xyz(0:3), L_r6d(3:9), R_xyz(9:12), R_r6d(12:18)]
        return row[0:3], row[3:9], row[9:12], row[12:18]

    def proj_pt(P):
        uv = project_point_fisheye_aria_np(P, cam_params, image_size)
        if uv is None:
            return None
        u, v = uv
        if rotate90:
            u, v = rotate_90_clockwise(u, v, W, H)
        return (int(u), int(v))

    # Build UV and rotation arrays
    uvL, uvR = [], []
    PL, PR, RL, RR = [], [], [], []

    for i in range(T):
        lpos, lr6d, rpos, rr6d = split_row(pred_seq_Tx18[i])
        PL.append(lpos)
        PR.append(rpos)

        # Convert rotation_6d to matrix using quaternion
        ql_wxyz = tf.forward(lr6d)
        ql_xyzw = np.array([ql_wxyz[1], ql_wxyz[2], ql_wxyz[3], ql_wxyz[0]])
        RL.append(R.from_quat(ql_xyzw).as_matrix())

        qr_wxyz = tf.forward(rr6d)
        qr_xyzw = np.array([qr_wxyz[1], qr_wxyz[2], qr_wxyz[3], qr_wxyz[0]])
        RR.append(R.from_quat(qr_xyzw).as_matrix())

        uvL.append(proj_pt(lpos))
        uvR.append(proj_pt(rpos))

    PL = np.asarray(PL)
    PR = np.asarray(PR)
    RL = np.asarray(RL)
    RR = np.asarray(RR)

    def draw_hand(uv_seq, P_seq, R_seq, color_bgr):
        # Draw trajectory lines
        for i in range(len(uv_seq) - 1):
            if uv_seq[i] is None or uv_seq[i + 1] is None:
                continue
            p1 = uv_seq[i]
            p2 = uv_seq[i + 1]
            # Fade color from deep to lighter along the path
            prog = i / max(len(uv_seq) - 1, 1)
            fade = 1.0 - 0.7 * prog  # near=1.0, far=0.3
            faded = tuple(int(c * fade + 255 * (1 - fade)) for c in color_bgr)
            cv2.line(canvas, p1, p2, faded, FUTURE_LINE_WIDTH)

        # Draw green +X arrows at up to 6 uniformly spaced samples
        n_ar = min(len(uv_seq), 6)
        step = max(1, len(uv_seq) // n_ar) if n_ar > 0 else 1
        for i in range(0, len(uv_seq), step):
            if uv_seq[i] is None:
                continue
            origin3d = P_seq[i]
            Rm = R_seq[i]
            end3d = origin3d + Rm[:, 0] * axis_len_m
            uv_end = project_point_fisheye_aria_np(end3d, cam_params, image_size)
            if uv_end is None:
                continue
            u0, v0 = uv_seq[i]
            u1, v1 = uv_end
            if rotate90:
                u1, v1 = rotate_90_clockwise(u1, v1, W, H)
            cv2.arrowedLine(
                canvas, (u0, v0), (int(u1), int(v1)), DARK_GREEN_BGR, 5, tipLength=0.3
            )

    # Draw both hand trajectories
    draw_hand(uvL, PL, RL, BLUE)
    draw_hand(uvR, PR, RR, RED)

    # Draw yellow circle at contact time point
    if (
        contact_t is not None
        and draw_contact
        and 0 <= contact_t < len(uvL)
        and 0 <= contact_t < len(uvR)
    ):
        # Draw for left hand
        if uvL[contact_t] is not None:
            cv2.circle(canvas, uvL[contact_t], 24, YELLOW, -1)
            cv2.circle(canvas, uvL[contact_t], 28, BLACK, 4)

        # Draw for right hand
        if uvR[contact_t] is not None:
            cv2.circle(canvas, uvR[contact_t], 24, YELLOW, -1)
            cv2.circle(canvas, uvR[contact_t], 28, BLACK, 4)

    return canvas


def create_3d_view_image_cam_xyz(
    P_L,
    R_L,
    P_R,
    R_R,
    width,
    height,
    axis_len_m=DEFAULT_AXIS_LEN_M,
    figsize_scale=1.5,
    elev_deg=45.0,
    azim_deg=-45.0,
):
    """
    Create 3D visualization in camera space.

    Args:
        P_L: (T, 3) left hand positions
        R_L: (T, 3, 3) left hand rotation matrices
        P_R: (T, 3) right hand positions
        R_R: (T, 3, 3) right hand rotation matrices
        width, height: Canvas dimensions
        axis_len_m: Length of rotation axes
        figsize_scale: Scale factor for figure size
        elev_deg, azim_deg: 3D view angles

    Returns:
        BGR image of 3D plot
    """
    fig = plt.figure(
        figsize=(figsize_scale * width / 100, figsize_scale * height / 100), dpi=100
    )
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def plot_hand(P, R, color_name):
        if P is None or len(P) < 2:
            return
        P = np.asarray(P, dtype=np.float64)
        # Plot thick line with gentle fade
        for i in range(len(P) - 1):
            prog = i / max(len(P) - 1, 1)
            fade = 1.0 - 0.7 * prog
            if color_name == "blue":
                rgb = (0, 0, 1.0)
            else:
                rgb = (1.0, 0, 0)
            col = tuple(np.array(rgb) * fade + (1 - fade))
            ax.plot(
                P[i : i + 2, 0],
                P[i : i + 2, 1],
                P[i : i + 2, 2],
                color=col,
                linewidth=14,
            )

        # Green +X arrows at up to 6 samples
        n_ar = min(len(P), 6)
        step = max(1, len(P) // n_ar) if n_ar > 0 else 1
        for i in range(0, len(P), step):
            d = R[i][:, 0]
            ax.quiver(
                P[i, 0],
                P[i, 1],
                P[i, 2],
                d[0],
                d[1],
                d[2],
                length=axis_len_m,
                color=(0.0, 0.6, 0.0),
                arrow_length_ratio=0.25,
                linewidth=4.0,
            )

    # Plot both hand trajectories in 3D
    plot_hand(P_L, R_L, "blue")
    plot_hand(P_R, R_R, "red")

    # Labels & view
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev_deg, azim=azim_deg)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.3)

    # Fit bounds with small padding
    allP = []
    if P_L is not None and len(P_L):
        allP.append(P_L)
    if P_R is not None and len(P_R):
        allP.append(P_R)

    if len(allP):
        A = np.concatenate(allP, axis=0)
        mn = A.min(0)
        mx = A.max(0)
        pad = 0.15 * max((mx - mn).max(), 1e-3)
        ax.set_xlim(mn[0] - pad, mx[0] + pad)
        ax.set_ylim(mn[1] - pad, mx[1] + pad)
        ax.set_zlim(mn[2] - pad, mx[2] + pad)

        # Make grid sparser
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    else:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.4)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def hstack_two(left_panel, right_panel, title_text=""):
    """Stack two panels horizontally with optional title."""
    h = max(left_panel.shape[0], right_panel.shape[0])

    def pad_h(img, H):
        if img.shape[0] == H:
            return img
        pad = H - img.shape[0]
        top = pad // 2
        bot = pad - top
        return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)

    combined = np.hstack([pad_h(left_panel, h), pad_h(right_panel, h)])

    if title_text:
        title_h = 56
        bar = np.ones((title_h, combined.shape[1], 3), np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.1
        thick = 3
        size = cv2.getTextSize(title_text, font, scale, thick)[0]
        x = (combined.shape[1] - size[0]) // 2
        y = (title_h + size[1]) // 2
        cv2.putText(bar, title_text, (x, y), font, scale, BLACK, thick)
        combined = np.vstack([bar, combined])

    return combined


# ----------------------------
# Main visualization function
# ----------------------------
def visualize_predictions(
    img_dir,
    result_pkl_path,
    output_dir,
    cam_params_dict=None,
    image_base_path=None,
    K=3,
):
    """
    Visualize predicted hand trajectories from inference results.

    Args:
        result_pkl_path: Path to pkl file from inference_new_stage2_od_6dof_fast_robohack.py
        output_dir: Directory to save visualization images
        cam_params_dict: Dict of camera parameters(optional)
        image_base_path: Base path for images (optional)
    """
    # Load results
    print(f"Loading results from {result_pkl_path}...")
    if not os.path.exists(result_pkl_path):
        print(f"Error: Result pkl not found at {result_pkl_path}")
        return

    all_data = pickle.load(open(result_pkl_path, "rb"))
    print(f"Loaded {len(all_data)} samples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    count = 0
    for cur in tqdm(all_data, desc="Processing samples"):
        if "ori_pred_wrist_pose" not in cur:
            print(f"Skipping sample - no ori_pred_wrist_pose field")
            continue

        # Get predicted trajectories (K, T, 18)
        ori_pred_wrist_pose = cur["ori_pred_wrist_pose"]
        if isinstance(ori_pred_wrist_pose, list):
            ori_pred_wrist_pose = np.array(ori_pred_wrist_pose)

        K = min(K, ori_pred_wrist_pose.shape[0])  # Number of predictions
        print(f"\nSample has {K} predicted trajectories")
        img_path = os.path.join(
            img_dir,
            cur["image"],
        )
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Get camera parameters
        if cam_params_dict:
            sample_id = cur["image"][:-4]
            if sample_id in cam_params_dict:
                cam_cfg = cam_params_dict[sample_id]
                cam_params = cam_cfg[0]
                image_size = cam_cfg[1]
            else:
                print(f"No camera params for sample_id: {sample_id}")
                continue
        else:
            # Use default camera parameters
            print("Using default camera parameters")
            cam_params = [500, 704, 704] + [0] * 12  # Default fisheye params
            image_size = (1408, 1408)

        # Load image
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Get intention text for title
        intention = cur.get("intention", "predicted trajectory")

        # Extract contact and end time if available
        contact_t = None
        end_t = None
        if "pred_contact" in cur and cur["pred_contact"] is not None:
            contact_t = round(cur["pred_contact"][0] * 4.5 * 10)
        if "pred_end" in cur and cur["pred_end"] is not None:
            end_t = round(cur["pred_end"][0] * 4.5 * 10)

        print(f"Intention: {intention} - contact_t={contact_t} - end_t={end_t}")

        # Visualize each of the K predictions
        for k in range(K):
            # Limit trajectory to end time if available
            if end_t is not None and end_t > 0:
                pred_seq = ori_pred_wrist_pose[k][:end_t]  # (T, 18)
            else:
                pred_seq = ori_pred_wrist_pose[k]  # (T, 18)

            T = pred_seq.shape[0]

            if T < 2:
                print(f"Skipping K={k} - trajectory too short")
                continue

            # Create overlay on RGB image with contact point marker
            overlay_panel = overlay_traj_on_rgb(
                rgb.copy(),
                pred_seq,
                cam_params,
                image_size,
                ROTATE_PIXELS_90CW,
                contact_t=contact_t,  # Pass contact time for yellow circle
            )

            # Build 3D visualization data
            def split_row(row):
                return row[0:3], row[3:9], row[9:12], row[12:18]

            PL, PR, RL, RR = [], [], [], []
            for i in range(T):
                lpos, lr6d, rpos, rr6d = split_row(pred_seq[i])
                PL.append(lpos)
                PR.append(rpos)

                # Convert rotation_6d to matrix using quaternion
                ql_wxyz = tf.forward(lr6d)
                ql_xyzw = np.array([ql_wxyz[1], ql_wxyz[2], ql_wxyz[3], ql_wxyz[0]])
                RL.append(R.from_quat(ql_xyzw).as_matrix())

                qr_wxyz = tf.forward(rr6d)
                qr_xyzw = np.array([qr_wxyz[1], qr_wxyz[2], qr_wxyz[3], qr_wxyz[0]])
                RR.append(R.from_quat(qr_xyzw).as_matrix())

            PL = np.asarray(PL)
            PR = np.asarray(PR)
            RL = np.asarray(RL)
            RR = np.asarray(RR)

            # Create 3D view
            view_3d_panel = create_3d_view_image_cam_xyz(
                PL,
                RL,
                PR,
                RR,
                width=rgb.shape[1],
                height=rgb.shape[0],
                axis_len_m=DEFAULT_AXIS_LEN_M,
                figsize_scale=1.5,
                elev_deg=20.0,
                azim_deg=125.0,
            )

            # Combine panels
            title = f"{intention} - Sample {k+1}/{K}"
            combo = hstack_two(overlay_panel, view_3d_panel, title_text=title)

            # Save image
            sample_name = os.path.splitext(os.path.basename(img_path))[0]
            out_name = f"{sample_name}_k{k+1}.jpg"
            out_full = os.path.join(output_dir, out_name)
            cv2.imwrite(out_full, combo)
            print(f"Saved: {out_full}")
            count += 1

    print(f"\nVisualization complete! Saved {count} images to {output_dir}")


if __name__ == "__main__":
    video_id = "examples"
    model_name = "EgoMAN-7B"
    img_dir = f"data/{video_id}"

    result_pkl_path = f"output/{model_name}-{video_id}.pkl"
    output_dir = f"output/visualizations/{video_id}"

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
    )
