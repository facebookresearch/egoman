# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.half_dim = dim // 2
        self.max_period = max_period

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        emb = math.log(self.max_period) / (self.half_dim - 1)
        emb = torch.exp(
            torch.arange(self.half_dim, device=t.device, dtype=t.dtype) * -emb
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def rot6d_to_matrix(x, eps=1e-6):  # x: (...,6)
    """
    Convert 6D rotation representation to 3x3 rotation matrix robustly.
    x: (..., 6)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    # Normalize first axis
    a1 = F.normalize(a1, dim=-1, eps=eps)

    # Remove projection of a2 on a1, then renormalize
    b2 = a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1
    b2 = F.normalize(b2, dim=-1, eps=eps)

    # Cross product for orthogonal third axis
    a3 = torch.cross(a1, b2, dim=-1)

    # Stack into rotation matrix (...,3,3)
    R = torch.stack((a1, b2, a3), dim=-2)

    # Optional re-orthogonalization via SVD for numerical safety
    # (slower, so only if persistent NaN)
    # U, _, Vt = torch.linalg.svd(R)
    # R = torch.matmul(U, Vt)

    return R


def geodesic_loss(R_pred, R_gt, reduce="mean"):
    """
    Stable geodesic (angular) loss between rotation matrices.
    """
    RtR = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    tr = RtR.diagonal(dim1=-2, dim2=-1).sum(-1)
    # Clamp trace inside valid SO(3) range
    cos_theta = (tr.clamp(-1.0, 3.0) - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos_theta)
    if reduce == "mean":
        return ang.mean()
    return ang


class FMTransformerDecoder6DoF(nn.Module):
    def __init__(
        self,
        cond_dim: int = 768,
        action_dim: int = 18,
        hidden_dim: int = 768,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        # Input projections
        self.hidden_dim = hidden_dim
        self.vis_proj = nn.Linear(1024, 768)
        self.text_proj = nn.Linear(768, 768)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.noisy_proj = nn.Linear(action_dim, hidden_dim)

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        # self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.time_proj = nn.Sequential(  # E -> 2*D
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.time_proj[-1].weight)  # start as identity
        nn.init.zeros_(self.time_proj[-1].bias)

        self.query_pos_emb = nn.Embedding(56, hidden_dim)

        # Modal embeddings to distinguish between trajectory and text modalities
        self.modal_emb = nn.Embedding(3, hidden_dim)  # 0: trajectory, 1: text

        # Full attention encoder for [hs_t, prev_traj]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Self-attention layer for noisy future before cross-attention
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, activation="gelu"
        )
        self.self_attention = nn.TransformerEncoder(self_attn_layer, num_layers=2)

        # Decoder with cross-attention to encoded context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 6)
        self.rot_output_proj = nn.Linear(hidden_dim, 12)

        # Layer norms
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def encode_context(
        self,
        hs_t: torch.FloatTensor,
        prev_traj: torch.FloatTensor,
        sample_t: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        timestamp=None,
    ):
        """
        Encode [prev_traj, text_emb] with full attention

        Args:
            hs_t: (B, cond_dim) - text embeddings from film
            prev_traj: (B, L_prev, action_dim) - previous trajectory
            sample_t: (B,) - time step
            mask: (B, L_prev + 1) - attention mask for padding

        Returns:
            encoded_context: (B, L_prev + 1, hidden_dim)
        """
        B = hs_t.shape[0]

        if prev_traj.shape[-1] != self.hidden_dim:
            prev_traj_proj = self.input_proj(prev_traj)  # (B, L_prev, hidden_dim)
        else:
            prev_traj_proj = prev_traj

        # Project text embeddings and add sequence dimension
        text_emb_proj = self.cond_proj(hs_t)  # (B, 1, hidden_dim)
        # Concatenate prev_traj and text embeddings
        context = torch.cat(
            [text_emb_proj, prev_traj_proj], dim=1
        )  # (B, L_prev + 1, hidden_dim)

        L_total = context.shape[1]

        # Add position embeddings
        pos_ids = (
            torch.arange(L_total, device=context.device).unsqueeze(0)
            - text_emb_proj.shape[1]
        )  # (1, L_total)
        pos_ids = pos_ids.repeat(B, 1)
        if timestamp is not None:
            pos_ids[:, -3:] = (
                timestamp.long() + 5
            )  # timestamp for start/end contact locations
        pos_ids[:, : text_emb_proj.shape[1]] = (
            0  # Text modality ID for the last position (text embeddings)
        )
        pos_emb = self.query_pos_emb(pos_ids)  # (B, L_total, hidden_dim)
        pos_emb = pos_emb.clone()  #
        pos_emb[:, : text_emb_proj.shape[1]] = 0
        # Add modal embeddings to distinguish between trajectory and text modalities
        modal_ids = torch.zeros(B, L_total, device=context.device, dtype=torch.long)
        modal_ids[:, : text_emb_proj.shape[1]] = torch.arange(
            1, text_emb_proj.shape[1] + 1
        )  # Text modality ID for the last position (text embeddings)
        modal_emb = self.modal_emb(modal_ids)  # (B, L_total, hidden_dim)

        context = context + pos_emb + modal_emb

        # Apply layer norm
        context = self.encoder_norm(context)

        # Encode with full attention
        if mask is not None:
            # mask should be (B, L_total) where True means padding
            encoded_context = self.encoder(
                context, src_key_padding_mask=mask[:, : context.shape[1]]
            )
        else:
            encoded_context = self.encoder(context)

        return encoded_context

    def decode_future(
        self,
        noisy_future: torch.FloatTensor,
        encoded_context: torch.FloatTensor,
        sample_t: torch.FloatTensor,
        hs_t: torch.FloatTensor,
        mask: torch.BoolTensor = None,
    ):
        """
        Decode noisy_future using encoded context as key/value

        Args:
            noisy_future: (B, L_future, action_dim) - noisy future trajectory as query
            encoded_context: (B, L_context, hidden_dim) - encoded [hs_t, prev_traj]
            sample_t: (B,) - time step
            hs_t: (B, cond_dim) - verb embeddings for FiLM conditioning

        Returns:
            decoded_future: (B, L_future, action_dim)
        """
        B, L_future = noisy_future.shape[:2]

        # Project noisy future
        noisy_future_proj = self.noisy_proj(noisy_future)  # (B, L_future, hidden_dim)

        # Add time embedding
        gamma, beta = (
            self.time_proj(self.time_emb(sample_t)).unsqueeze(1).chunk(2, -1)
        )  # (B, 1, hidden_dim)
        gamma = gamma.clamp(-3, 3)
        beta = beta.clamp(-3, 3)

        x = F.layer_norm(noisy_future_proj, (noisy_future_proj.size(-1),))
        noisy_future_proj = x * (1.0 + gamma) + beta  # (B,T,D)

        # Add position embeddings for future
        pos_ids = (
            torch.arange(L_future, device=noisy_future.device).unsqueeze(0) + 5
        )  # (1, L_future)
        pos_emb = self.query_pos_emb(pos_ids)  # (1, L_future, hidden_dim)
        modal_ids = torch.zeros(
            B, L_future, device=noisy_future.device, dtype=torch.long
        )
        modal_emb = self.modal_emb(modal_ids)

        noisy_future_proj = noisy_future_proj + pos_emb + modal_emb

        # Apply layer norm
        noisy_future_proj = self.decoder_norm(noisy_future_proj)

        noisy_future_proj = noisy_future_proj + self.self_attention(
            noisy_future_proj,
            src_key_padding_mask=mask[:, -L_future:] if mask is not None else None,
        )

        future_mask = mask[:, -L_future:] if mask is not None else None
        context_mask = mask[:, :-L_future] if mask is not None else None
        # Decode with cross-attention to encoded context
        decoded = self.decoder(
            tgt=noisy_future_proj,
            memory=encoded_context,
            tgt_key_padding_mask=future_mask,
            memory_key_padding_mask=context_mask,
        )

        # Project to output dimension
        pos_output = self.output_proj(decoded)
        rot_output = self.rot_output_proj(decoded)
        B, N, C = pos_output.shape
        output = torch.cat(
            [pos_output.reshape(B, N, 2, -1), rot_output.reshape(B, N, 2, -1)], dim=-1
        ).reshape(B, N, -1)

        return output, future_mask

    def model_forward(
        self,
        hs_t: torch.FloatTensor,
        prev_traj: torch.FloatTensor,
        noisy_future: torch.FloatTensor,
        sample_t: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        timestamp=None,
    ):
        """
        Full forward pass: encode context then decode future

        Args:
            hs_t: (B, L_hs, cond_dim) - variable length but consistent in batch
            prev_traj: (B, L_prev, action_dim) - previous trajectory
            noisy_future: (B, L_future, action_dim) - noisy future trajectory
            sample_t: (B,) - time step
            context_mask: (B, L_hs + L_prev) - mask for context
            future_mask: (B, L_future) - mask for future

        Returns:
            output: (B, L_future, action_dim) - predicted velocity
        """
        # Encode context [hs_t, prev_traj]
        encoded_context = self.encode_context(
            hs_t, prev_traj, sample_t, mask, timestamp
        )

        # Decode future using encoded context
        output, future_mask = self.decode_future(
            noisy_future, encoded_context, sample_t, hs_t, mask
        )

        return output, future_mask

    def forward(
        self,
        hs_t: torch.FloatTensor,
        prev_traj: torch.FloatTensor,
        gt_traj: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        loss: bool = False,
        timestamp=None,
    ):
        """
        Training forward pass with flow matching

        Args:
            hs_t: (B, L_hs, cond_dim) - variable length but consistent in batch
            prev_traj: (B, L_prev, action_dim) - previous trajectory
            gt_traj: (B, L_future, action_dim) - ground truth future trajectory
            loss: bool - whether to compute loss

        Returns:
            If loss=False: (pred_v, target_v, sample_t)
            If loss=True: (pred_v, target_v, sample_t, loss_value)
        """
        B, L_future, action_dim = gt_traj.shape
        device = gt_traj.device

        # Sample time step
        sample_t = torch.rand(B, device=device)

        # Flow matching noise schedule
        noise = torch.randn_like(gt_traj)
        noisy_future = (1 - sample_t[:, None, None]) * noise + sample_t[
            :, None, None
        ] * gt_traj
        # Forward pass
        pred_v, future_mask = self.model_forward(
            hs_t, prev_traj, noisy_future, sample_t, mask, timestamp=timestamp
        )

        # Target velocity
        target_v = gt_traj - noise

        if not loss:
            return pred_v, target_v, sample_t
        else:
            loss_value = self.compute_loss(
                pred_v, target_v, future_mask, prev_traj, noise, sample_t
            )

            return pred_v, target_v, sample_t, loss_value

    def compute_loss(
        self,
        pred,
        target,
        future_mask=None,
        prev_traj=None,
        noisy_future=None,
        sample_t=None,
        traj_weight=1.0,
        rot_weight=0.5,
        smooth_weight=0.1,
        # smooth_weight=0.0,
    ):
        """
        Compute MSE loss with optional masking for valid positions only

        Args:
            pred: (B, L, action_dim) - predicted velocity
            target: (B, L, action_dim) - target velocity
            future_mask: (B, L) - mask where 0 means valid, 1 means invalid/padding

        Returns:
            loss: scalar tensor
        """
        if future_mask is not None:
            valid = (future_mask == 0).float().unsqueeze(-1)  # (B,L,1) 1=valid
        else:
            valid = torch.ones_like(pred[..., :1])

        # Translation loss on valid steps; average by valid *dims
        L_xyz_pred = torch.cat([pred[..., :3], pred[..., 9:12]], dim=-1)
        L_xyz_tgt = torch.cat([target[..., :3], target[..., 9:12]], dim=-1)
        trans_w = valid.expand_as(L_xyz_pred)
        trans_loss = F.mse_loss(
            L_xyz_pred * trans_w, L_xyz_tgt * trans_w, reduction="sum"
        )
        trans_den = trans_w.sum().clamp_min(1.0)
        trans_loss = trans_loss / trans_den

        # Rotation loss: geodesic + MSE on rot6d
        left_pred = pred[..., 3:9]
        right_pred = pred[..., 12:18]
        left_tgt = target[..., 3:9]
        right_tgt = target[..., 12:18]

        # Geodesic loss on rotation matrices
        Rl_pred = rot6d_to_matrix(left_pred)
        Rr_pred = rot6d_to_matrix(right_pred)
        Rl_tgt = rot6d_to_matrix(left_tgt)
        Rr_tgt = rot6d_to_matrix(right_tgt)

        # weight angles by valid mask (broadcast to (...,1,1))
        w3x3 = valid.unsqueeze(-1)
        rot_loss_l = geodesic_loss(Rl_pred * w3x3 + (1 - w3x3) * Rl_tgt, Rl_tgt)
        rot_loss_r = geodesic_loss(Rr_pred * w3x3 + (1 - w3x3) * Rr_tgt, Rr_tgt)
        rot_geodesic_loss = 0.0 * (rot_loss_l + rot_loss_r)

        rot6d_pred = torch.cat([left_pred, right_pred], dim=-1)  # (B, L, 12)
        rot6d_tgt = torch.cat([left_tgt, right_tgt], dim=-1)  # (B, L, 12)
        rot_w = valid.expand_as(rot6d_pred)
        rot_mse_loss = F.mse_loss(
            rot6d_pred * rot_w, rot6d_tgt * rot_w, reduction="sum"
        )
        rot_mse_den = rot_w.sum().clamp_min(1.0)
        rot_mse_loss = rot_mse_loss / rot_mse_den

        # Combined rotation loss
        rot_loss = rot_geodesic_loss + rot_mse_loss

        loss = traj_weight * trans_loss + rot_weight * rot_loss

        return loss

    @torch.no_grad()
    def sample_trajectory(
        self,
        hs_t,
        prev_traj,
        L_future,
        action_dim,
        device,
        n_steps=50,
        mask=None,
        timestamp=None,
    ):
        """
        Sample trajectory using the trained model

        Args:
            hs_t: (B, L_hs, cond_dim) - context features
            prev_traj: (B, L_prev, action_dim) - previous trajectory
            L_future: int - length of future trajectory to sample
            action_dim: int - action dimension
            device: torch.device
            n_steps: int - number of sampling steps
            context_mask: (B, L_hs + L_prev) - mask for context

        Returns:
            sampled_trajectory: (B, L_future, action_dim) - sampled trajectories for all batch samples
        """
        B = hs_t.shape[0]
        x = torch.randn(B, L_future, action_dim, device=device)

        # Uniform step size
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = (step + 0.5) / n_steps  # (0,1]
            t = torch.full((B,), t_val, device=device)
            v, _ = self.model_forward(hs_t, prev_traj, x, t, mask, timestamp=timestamp)
            # Integrate forward in time
            x = x + dt * v

        return x  # Return all batch samples
