# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import random

import _init_paths

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset.dataloaders.motion_pretrain_dataset import (
    EgoMANDataset,
    seq_collate_egoman,
)
from decord import VideoReader
from model.egoman_model import FMTransformerDecoder6DoF
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

local_rank = None


def compute_masked_ade(
    distances: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    distances: (B, 2, L) L2 distance of positions per hand,timestep
    valid_mask: (B, 2, L) boolean/0-1 mask of valid future steps
    returns scalar ADE (mean distance over all valid entries)
    """
    # ensure same dtype/device
    valid = valid_mask.to(distances.dtype).unsqueeze(1)
    denom = valid.sum().clamp_min(1.0)
    return (distances * valid).sum() / denom


def set_seed(seed: int, deterministic: bool = False, rank: int = 0):
    """
    Seed Python/NumPy/PyTorch. If deterministic=True, enable stricter
    determinism (slower; may error on nondeterministic ops).
    In DDP, pass the global rank so each worker can derive per-worker seeds.
    """
    # Base seeds (same across ranks)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: per-rank bump so dataloaders/augmentations differ across ranks
    worker_seed = (seed + rank) % (2**31 - 1)
    torch.random.manual_seed(worker_seed)

    if deterministic:
        # cuDNN & general determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Error (or warn) on nondeterministic ops
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            torch.set_deterministic_debug_mode("warn")

        # cuBLAS matmul determinism (needed on some GPUs/versions)
        # Choose one of the two configs:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # or ":4096:8"

    # (Optional) reduce floating jitter on matmul kernels in fp32
    # torch.set_float32_matmul_precision("high")  # default is "highest" on some builds


def seed_worker(worker_id: int):
    """
    Use with DataLoader(worker_init_fn=seed_worker, generator=...)
    Ensures each worker has a distinct, repeatable RNG state.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class MotionExpertTrainer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.fm_model = FMTransformerDecoder6DoF()

    def forward(self, batch):
        pixel_values_videos = batch["pixel_values_videos"]
        text_feature = batch["text_feature"].float()
        pre_traj = batch["past_traj"]
        gt_traj = batch["fut_traj"]
        pre_quat = batch["past_quat"]
        gt_quat = batch["fut_quat"]
        gt_traj = torch.cat([gt_traj, gt_quat], dim=-1)
        pre_traj = torch.cat([pre_traj, pre_quat], dim=-1)
        gt_traj = gt_traj.permute(0, 2, 1, 3).flatten(2, 3)
        pre_traj = pre_traj.permute(0, 2, 1, 3).flatten(2, 3)
        pre_quat = pre_quat.permute(0, 2, 1, 3).flatten(2, 3)
        gt_quat = gt_quat.permute(0, 2, 1, 3).flatten(2, 3)
        future_valid = batch["future_valid"]
        start_contact_loc = batch["start_contact_loc"]
        timestamp = batch["start_end_timestamp"]

        context_embeds = pixel_values_videos
        text_embeds = text_feature
        self.fm_model.to(torch.float32)

        pre_traj = torch.cat([pre_traj, start_contact_loc], dim=1)
        context_embeds = (
            self.fm_model.vis_proj(context_embeds.squeeze(1))
            + self.fm_model.text_proj(text_embeds.squeeze(1))
        ).unsqueeze(1)

        pred_v, target_v, sample_t, fm_loss = self.fm_model(
            hs_t=context_embeds.to(torch.float32),  # Context features
            prev_traj=pre_traj.to(torch.float32),  # Previous trajectory
            gt_traj=gt_traj.to(torch.float32),  # Ground truth future trajectory
            mask=1 - future_valid.to(torch.float32),  # No future mask for now
            loss=True,  # Compute loss
            timestamp=timestamp,
        )

        return fm_loss

    @torch.no_grad()
    def sample(self, batch, n_steps=50):
        pixel_values_videos = batch["pixel_values_videos"]
        text_feature = batch["text_feature"].float()
        pre_traj = batch["past_traj"]
        gt_traj = batch["fut_traj"]
        pre_quat = batch["past_quat"]
        gt_quat = batch["fut_quat"]
        gt_traj = torch.cat([gt_traj, gt_quat], dim=-1)
        pre_traj = torch.cat([pre_traj, pre_quat], dim=-1)
        gt_traj = gt_traj.permute(0, 2, 1, 3).flatten(2, 3)
        pre_traj = pre_traj.permute(0, 2, 1, 3).flatten(2, 3)
        pre_quat = pre_quat.permute(0, 2, 1, 3).flatten(2, 3)
        gt_quat = gt_quat.permute(0, 2, 1, 3).flatten(2, 3)
        future_valid = batch["future_valid"]
        start_contact_loc = batch["start_contact_loc"]
        timestamp = batch["start_end_timestamp"]

        device = pixel_values_videos.device
        context_embeds = pixel_values_videos
        text_embeds = text_feature
        self.fm_model.to(torch.float32)

        pre_traj = torch.cat([pre_traj, start_contact_loc], dim=1)
        context_embeds = (
            self.fm_model.vis_proj(context_embeds.squeeze(1))
            + self.fm_model.text_proj(text_embeds.squeeze(1))
        ).unsqueeze(1)
        sampled_traj = self.fm_model.sample_trajectory(
            hs_t=context_embeds.to(torch.float32),  # Use entire batch
            prev_traj=pre_traj.to(torch.float32),  # Use entire batch
            L_future=50,
            action_dim=18,
            device=device,
            n_steps=n_steps,
            mask=1 - future_valid.to(torch.float32),  # Use entire batch
            timestamp=timestamp,
        )
        sampled_traj = sampled_traj.cpu().numpy()  # (B, L, 18) numpy

        return sampled_traj  # Returns (B, 50, 6) for all batch samples


def main(output_path, num_epochs=500):
    # DDP setup
    dist.init_process_group("nccl")
    global local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed(42, deterministic=True, rank=rank)

    # Model
    model = MotionExpertTrainer().to(device)

    # Synchronize after model initialization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Use DDP for distributed training
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    rank0_print("Model wrapped with DDP for distributed training")

    # Training Dataset
    train_dataset = EgoMANDataset(split="train")

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        sampler=train_sampler,
        collate_fn=seq_collate_egoman,
        num_workers=4,
    )

    # Validation Dataset
    val_dataset = EgoMANDataset(split="val")

    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=6,
        sampler=val_sampler,
        collate_fn=seq_collate_egoman,
        num_workers=4,
        shuffle=False,
    )

    # Optimizer
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        dataloader_tqdm = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
            disable=(local_rank != 0),
        )

        for step, batch in dataloader_tqdm:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if local_rank == 0 and step % 10 == 0:
                tqdm.write(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

        # Evaluation on validation set
        if local_rank == 0 and epoch % 5 == 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                # Evaluate on validation set
                val_tqdm = tqdm(
                    enumerate(val_dataloader),
                    total=min(10, len(val_dataloader)),  # Evaluate on first 10 batches
                    desc=f"Validation Epoch {epoch}",
                    disable=(local_rank != 0),
                )

                for val_step, val_batch in val_tqdm:
                    if val_step >= 10:  # Limit validation to 10 batches for speed
                        break

                    # Move batch to device
                    for key in val_batch:
                        if isinstance(val_batch[key], torch.Tensor):
                            val_batch[key] = val_batch[key].to(device)
                    val_loss = model(val_batch)
                    val_losses.append(val_loss.item())

                # Calculate average validation loss
                if val_losses:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    rank0_print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")

                eval_batch = next(iter(val_dataloader))
                for key in eval_batch:
                    if isinstance(eval_batch[key], torch.Tensor):
                        eval_batch[key] = eval_batch[key].to(device)

                # Sample trajectories
                sampled_traj = model.module.sample(eval_batch)  # (B, L, 18) numpy
                sampled_traj_tensor = torch.from_numpy(sampled_traj).to(
                    device
                )  # (B, L, 18)

                # Ground truth (positions+rot6d) & valid mask
                gt_pos = eval_batch["fut_traj"]  # (B, 2, L, 3)
                future_valid = eval_batch["future_valid"]  # (B, 2, L_total)
                # Use the same slice as training (you used 4:) to align with the predicted horizon
                future_valid = future_valid[..., 9:]  # -> (B, 2, L)
                L_future = gt_pos.shape[2]

                # Reshape sampled to (B, 2, L, 9): [xyz(3)+rot6d(6)] per hand
                sampled_traj_reshaped = sampled_traj_tensor.view(
                    sampled_traj_tensor.size(0), L_future, 2, 9
                ).permute(0, 2, 1, 3)

                # --- ADE over positions ---
                # L2 distances per (B, 2, L)
                distances = torch.norm(
                    sampled_traj_reshaped[..., :3] - gt_pos[..., :3], dim=3
                )  # (2, 50)

                ade = compute_masked_ade(distances, future_valid)
                rank0_print(f"[Epoch {epoch}] Validation ADE: {ade.item():.4f}")

        # Save checkpoint
        if local_rank == 0 and epoch % 5 == 0:
            torch.save(
                model.module.fm_model.state_dict(),
                output_path,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    folder = "../data/weights"
    os.makedirs(folder, exist_ok=True)
    output_path = f"{folder}/fm_motion_model.pth"

    main(output_path, num_epochs=250)
