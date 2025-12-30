# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_normalize(v, dim=-1, eps=1e-6):
    # Forward: identical to normalize for large norms
    # Backward: avoids 1/||v|| when ||v|| ~ 0 by replacing with a fixed unit basis (zero grad wrt v there)
    n = v.norm(dim=dim, keepdim=True)
    ok = n > eps
    v_unit = torch.zeros_like(v)
    # pick a deterministic unit vector to avoid NaN grads
    v_unit[..., 0] = 1.0
    out = torch.where(ok, v / n.clamp_min(eps), v_unit)
    # stop gradient through the replacement branch to avoid polluting grads
    out = torch.where(ok, out, out.detach())
    return out


def rot6d_to_matrix_safe(x, eps=1e-6):
    with torch.autocast(device_type=torch.device("cuda").type, enabled=False):
        x = x.float()
        a1 = safe_normalize(x[..., 0:3], eps=eps)
        a2 = x[..., 3:6]
        proj = (a1 * a2).sum(dim=-1, keepdim=True) * a1
        b2 = safe_normalize(a2 - proj, eps=eps)
        a3 = safe_normalize(torch.cross(a1, b2, dim=-1), eps=eps)
        # re-orthogonalize b2 to ensure right-handedness
        b2 = safe_normalize(torch.cross(a3, a1, dim=-1), eps=eps)
        R = torch.stack((a1, b2, a3), dim=-2)
        return R


def geodesic_angle_safe(R_pred, R_gt, reduce="none", eps=1e-8):
    with torch.autocast(device_type=torch.device("cuda").type, enabled=False):
        R_pred = R_pred.float()
        R_gt = R_gt.float()
        RtR = torch.matmul(R_gt.transpose(-1, -2), R_pred)
        trace = RtR.diagonal(dim1=-2, dim2=-1).sum(-1)
        c = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # stay off ±1
        skew = RtR - RtR.transpose(-1, -2)
        s = 0.5 * torch.linalg.vector_norm(
            torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1),
            dim=-1,
        )
        # keep away from (0,0) to avoid undefined gradient
        s = s + eps
        ang = torch.atan2(s, c)
        if reduce == "mean":
            return ang.mean()
        if reduce == "sum":
            return ang.sum()
        return ang


class SemanticAligner(nn.Module):
    """
    Aligns action hidden state (from <ACT>) with CLIP text embeddings via InfoNCE.

    Args:
        hidden_dim: size of act_hidden (e.g. 3584)
        proj_dim: size of CLIP text embedding (e.g. 768)
    """

    def __init__(self, hidden_dim: int, proj_dim: int = 768, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, act_hidden: torch.Tensor, gt_act_vec: torch.Tensor):
        pred_emb = self.proj(act_hidden)  # [N, proj_dim]
        pred_emb = F.normalize(pred_emb, dim=-1)
        gt_emb = F.normalize(gt_act_vec, dim=-1)

        valid_mask = (gt_act_vec != -1000).any(dim=-1)
        pred_valid = pred_emb[valid_mask]
        gt_valid = gt_emb[valid_mask]

        K = pred_valid.size(0)

        dummy_loss = pred_emb.sum() * 0.0  # touches weights

        if K == 0:
            return dummy_loss, pred_emb
        if K < 10:
            sim = F.cosine_similarity(pred_valid, gt_valid, dim=-1)
            act_loss = 1 - sim.mean()
        else:
            logits = (pred_valid @ gt_valid.T) / self.temperature
            labels = torch.arange(K, device=pred_valid.device, dtype=torch.long)
            act_loss = F.cross_entropy(logits, labels)

        return act_loss + dummy_loss, pred_emb

    def inference(self, act_hidden: torch.Tensor):
        pred_emb = self.proj(act_hidden)  # [N, proj_dim]
        pred_emb = F.normalize(pred_emb, dim=-1)
        return pred_emb


def huber_loss(pred, target, beta, mask=None, reduce="mean"):
    """
    Smooth-L1 (Huber) with optional mask.
    pred, target: same shape
    beta: threshold in same units as pred/target
    mask: same shape as pred/target (1=valid, 0=ignore)
    """
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        loss = loss.sum() / denom if reduce == "mean" else loss.sum()
    else:
        loss = loss.mean() if reduce == "mean" else loss.sum()
    return loss


class TrajDecoder(nn.Module):
    def __init__(self, hidden_dim=512, token_dim=512, with_2dloss=True):
        super().__init__()
        self.token_dim = token_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # scalar time
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),  # 6 coords: l_xyz + r_xyz
        )
        self.with_2dloss = with_2dloss
        if self.with_2dloss:
            self.coord2d_mlp = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 4),  # 4 coords: l_xy + r_xy
            )
        self.quat_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 12),  # 12 rotation 6dof
        )

    def comput_rot_loss(self, pred, gt, eps=1e-8):
        """
        Args:
            R_pred: (K, 3, 3)
            R_gt:   (K, 3, 3)

        Returns:
            rot_loss: scalar
        """
        # Compute per-element MSE
        with torch.autocast(device_type=torch.device("cuda").type, enabled=False):
            pred = pred.float()
            gt = gt.float()
            Lp, Rp = pred[..., :6], pred[..., 6:]
            Lt, Rt = gt[..., :6], gt[..., 6:]

            Lvalid = (Lt != -1000).all(dim=-1)
            Rvalid = (Rt != -1000).all(dim=-1)
            safe6 = pred.new_tensor([1, 0, 0, 0, 1, 0])

            Lt = torch.where(Lvalid.unsqueeze(-1), Lt, safe6)
            Rt = torch.where(Rvalid.unsqueeze(-1), Rt, safe6)

            Rl_pred = rot6d_to_matrix_safe(Lp)
            Rl_tgt = rot6d_to_matrix_safe(Lt)
            Rr_pred = rot6d_to_matrix_safe(Rp)
            Rr_tgt = rot6d_to_matrix_safe(Rt)

            ang_L = geodesic_angle_safe(Rl_pred, Rl_tgt)
            ang_R = geodesic_angle_safe(Rr_pred, Rr_tgt)

            # mask invalids BEFORE reduction
            ang_L = torch.where(Lvalid, ang_L, ang_L.new_zeros(()).expand_as(ang_L))
            ang_R = torch.where(Rvalid, ang_R, ang_R.new_zeros(()).expand_as(ang_R))

            wL, wR = Lvalid.float(), Rvalid.float()
            denom = (wL.sum() + wR.sum()).clamp_min(eps)
            loss = ((wL * ang_L).sum() + (wR * ang_R).sum()) / denom
            return loss

    def forward(
        self,
        hidden_states,
        ori_gt_traj=None,
        *,
        # Huber thresholds (tune for your units: meters; 0.07 ≈ 7 cm)
        beta_3d=0.07,
        beta_2d=0.02,  # in normalized coords (~1/50th of image width/height)
        beta_time=3.0,  # frames; L2-like within ±beta_time frames
        # Time-window (Gaussian) that modulates coord/quaternion loss by time proximity
        sigma_time=3.0,  # frames; larger = more tolerant
    ):
        """
        Args:
            hidden_states: (K, C)
            ori_gt_traj:   (K, 11+?) GT: [time, l_x,l_y,l_z, r_x,r_y,r_z, u_l,v_l, u_r,v_r, quat(12)]
                           invalid entries are -1000
        Returns:
            pred_time:   (K, 1)
            pred_coords: (K, 18)   [L_xyz(3),L_rot6d(6), R_xyz(3),R_rot6d(6)]
            total_loss:  scalar
        """
        # Predict heads
        with_2dloss = self.with_2dloss
        pred_time = self.time_mlp(hidden_states)  # (K,1)
        pred_coords = self.coord_mlp(hidden_states)  # (K,6)  [L_xyz(3), R_xyz(3)]
        if with_2dloss:
            pred_coords_2d = self.coord2d_mlp(
                hidden_states
            )  # (K,4) [u_l,v_l,u_r,v_r] (pixels)
        pred_quat = self.quat_mlp(hidden_states)  # (K,12) [L_rot6d(6), R_rot6d(6)]

        total_loss = None

        if ori_gt_traj is not None:
            gt = ori_gt_traj

            # -------------------------------
            # Masks for valid fields
            # -------------------------------
            time_gt = gt[..., 0:1]
            time_valid = (time_gt != -1000).float()

            xyz_gt = gt[..., 1:7]  # (K,6)
            xyz_valid = (xyz_gt != -1000).float()

            if with_2dloss:
                uv_gt = gt[..., 7:11]  # (K,4) pixels
                uv_valid = (uv_gt != -1000).float()

            # If quats provided after index 11
            has_quat = gt.size(-1) > 11
            if has_quat:
                quat_gt = gt[..., 11:]  # (K,12)
                quat_valid = (quat_gt != -1000).float()

            # -------------------------------
            # Time loss: Huber in frames
            # -------------------------------
            # Use Huber so errors within ±beta_time behave L2-like, beyond that L1-like.
            pred_time_loss = huber_loss(
                pred_time, time_gt, beta=beta_time, mask=time_valid
            )

            # -------------------------------
            # Time-window weight for coord/quat
            # -------------------------------
            # Compute a soft weight w_t in [~0,1], high when |pred_time - gt_time| is small.
            # w_t = exp( -0.5 * ((Δt) / sigma_time)^2 ); Δt in frames
            if time_valid.sum() > 0:
                dt = (pred_time - time_gt).abs()
                w_t = torch.exp(-0.5 * (dt / sigma_time) ** 2) * time_valid  # (K,1)
            else:
                # No valid time gt; fall back to weight 1 everywhere
                w_t = torch.ones_like(pred_time)

            # Broadcast to field sizes
            w_xyz = w_t.expand_as(xyz_gt)
            w_quat = (
                w_t.new_ones(pred_quat.shape)
                if not has_quat
                else w_t.expand_as(pred_quat)
            )

            # -------------------------------
            # 3D coord loss: Huber per-dim
            # -------------------------------
            pred_3d_loss = huber_loss(
                pred_coords, xyz_gt, beta=beta_3d, mask=xyz_valid * w_xyz
            )

            # -------------------------------
            # Optional 2D coord loss (normalize by image size internally)
            # -------------------------------
            if with_2dloss:
                # Normalize to [0,1] by width=1408; if you also have height, you can scale accordingly
                pred_uv_n = pred_coords_2d / 1408.0
                gt_uv_n = uv_gt / 1408.0
                pred_2d_loss = huber_loss(
                    pred_uv_n,
                    gt_uv_n,
                    beta=beta_2d,
                    mask=uv_valid * w_t.expand_as(uv_gt),
                )

            # -------------------------------
            # Quaternion losses (Huber on raw 6D reps + your geometric rot loss)
            # -------------------------------
            if has_quat:
                quat_huber = huber_loss(
                    pred_quat,
                    quat_gt,
                    beta=0.2,  # 0.2 is a mild Huber for normalized 6D reps
                    mask=quat_valid * w_quat,
                )
                # Geodesic/rotation consistency term (keep your existing function)
                rot_loss = 0.15 * self.comput_rot_loss(pred_quat, quat_gt)
            else:
                quat_huber = torch.tensor(0.0, device=hidden_states.device)
                rot_loss = torch.tensor(0.0, device=hidden_states.device)

            # -------------------------------
            # Combine with weights
            # -------------------------------
            # λ_time, λ_3d, λ_2d = 1.0, 1.0, 0.5
            λ_time, λ_3d, λ_2d = 1.0, 2.0, 0.5
            total_loss = λ_time * pred_time_loss + λ_3d * pred_3d_loss
            if with_2dloss:
                total_loss = total_loss + λ_2d * pred_2d_loss

            total_loss = total_loss + 0.5 * quat_huber + rot_loss

        # Pack output coords back to your (K,18) format: [L_xyz(3),L_rot6d(6), R_xyz(3),R_rot6d(6)]
        pred_coords_packed = torch.cat(
            [pred_coords.reshape(-1, 2, 3), pred_quat.reshape(-1, 2, 6)], dim=-1
        ).reshape(-1, 18)

        return pred_time, pred_coords_packed, total_loss

    @torch.no_grad()
    def inference(self, hidden_states):
        """Only forward pass without loss"""
        pred_time = self.time_mlp(hidden_states)
        pred_coords = self.coord_mlp(hidden_states)
        pred_quat = self.quat_mlp(hidden_states)
        pred_coords = torch.cat(
            [pred_coords.reshape(-1, 2, 3), pred_quat.reshape(-1, 2, 6)], dim=-1
        ).reshape(-1, 18)
        return torch.cat([pred_time, pred_coords], dim=-1)  # (K, 7)

    @torch.no_grad()
    def inference_stage2(self, hidden_states):
        """Only forward pass without loss"""
        pred_time = self.time_mlp(hidden_states)
        pred_coords = self.coord_mlp(hidden_states)
        pred_quat = self.quat_mlp(hidden_states)
        return torch.cat([pred_time, pred_coords, pred_quat], dim=-1)  # (K, 7)
