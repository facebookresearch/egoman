# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)

from .flow_matching_decoder import FMTransformerDecoder6DoF
from .hand_decoder import SemanticAligner, TrajDecoder


def all_gather_tensor(tensor):
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


@torch.no_grad()
def nearest_time_idx(vlm_t: torch.Tensor, gt_ts: torch.Tensor) -> torch.Tensor:
    """
    vlm_t: (B,) VLM timestamp (already mapped to the same units as gt_ts indices or seconds)
    gt_ts: (B, T) ground-truth per-step timestamps or index grid
    Returns: (B,) nearest indices (long)
    """
    # If gt_ts are indices [0..T-1], you can pass a prebuilt grid
    # We compute nearest by absolute difference
    diff = torch.abs(gt_ts - vlm_t.unsqueeze(-1))  # (B,T)
    idx = diff.argmin(dim=-1)  # (B,)
    return idx.long()


# REPLACE the previous `plausibility_gate` with this version (adds time check)
@torch.no_grad()
def plausibility_gate(
    vlm_wpt_xyz: torch.Tensor,  # (B,3)
    vlm_t: torch.Tensor,  # (B,)
    gt_traj_xyz: torch.Tensor,  # (B,T,3)
    gt_ts: torch.Tensor,  # (B,T)
    time_win: int = 5,
    dist_thr_m: float = 0.07,
    time_thr: float = 5.0,  # NEW: max allowed time gap (in frames or same unit as gt_ts)
):
    """
    Returns matched_idx(B,), plausible(B,) bool, min_dist(B,), time_ok(B,) bool
    A waypoint is 'plausible' ONLY IF distance<=dist_thr_m AND time_diff<=time_thr.
    """
    B, T, _ = gt_traj_xyz.shape
    idx0 = nearest_time_idx(vlm_t, gt_ts)  # (B,)
    matched_idx = torch.zeros(B, dtype=torch.long, device=gt_traj_xyz.device)
    min_dist = torch.full((B,), 1e9, device=gt_traj_xyz.device)
    dist_ok = torch.zeros(B, dtype=torch.bool, device=gt_traj_xyz.device)
    time_ok = torch.zeros(B, dtype=torch.bool, device=gt_traj_xyz.device)

    for b in range(B):
        lo = max(0, idx0[b].item() - time_win)
        hi = min(T, idx0[b].item() + time_win + 1)
        seg = gt_traj_xyz[b, lo:hi]  # (W,3)
        d = torch.norm(seg - vlm_wpt_xyz[b], dim=-1)  # (W,)
        j = torch.argmin(d)
        dmin = d[j]
        idx_match = lo + j
        matched_idx[b] = idx_match
        min_dist[b] = dmin
        dist_ok[b] = dmin <= dist_thr_m
        # time diff w.r.t. matched index
        tdiff = torch.abs(vlm_t[b] - gt_ts[b, idx_match])
        time_ok[b] = tdiff <= time_thr

    plausible = dist_ok & time_ok
    return matched_idx, plausible, min_dist, time_ok


@torch.no_grad()
def constrained_shift_xyz_time(
    gt_traj_xyz: torch.Tensor,  # (B, T, 3)
    gt_ts: torch.Tensor,  # (B, T)
    vlm_xyz_list: list,  # list of (B, 3): [contactL, contactR, endL, endR]
    vlm_t_list: list,  # list of (B,): [t_contactL, t_contactR, t_endL, t_endR]
    max_loc_shift_m: float = 0.07,
    max_time_shift: float = 5.0,
):
    """
    When plausible_left or plausible_right == 0,
    find the GT point most closely matching each VLM waypoint in space & time,
    then shift that GT point toward the VLM target within spatial/time constraints.

    Returns:
        shifted_xyz_list: list[(B, 3)] - shifted GT coords for each waypoint
        shifted_t_list: list[(B,)] - shifted GT times for each waypoint
        matched_idx_list: list[(B,)] - indices of matched GT points
    """
    shifted_xyz_list, shifted_t_list, matched_idx_list = [], [], []

    for vlm_xyz, vlm_t in zip(vlm_xyz_list, vlm_t_list):
        B, T, _ = gt_traj_xyz.shape

        # ---- find best-matched GT point per batch ----
        matched_idx = torch.zeros(B, dtype=torch.long, device=gt_traj_xyz.device)
        gt_xyz_match = torch.zeros_like(vlm_xyz)
        gt_t_match = torch.zeros_like(vlm_t)

        for b in range(B):
            # combine space + time similarity
            d_space = torch.norm(gt_traj_xyz[b] - vlm_xyz[b], dim=-1)  # (T,)
            d_time = torch.abs(gt_ts[b] - vlm_t[b])  # (T,)
            # normalize and combine for a total score
            score = d_space / d_space.max().clamp_min(1e-6) + 0.1 * (
                d_time / d_time.max().clamp_min(1e-6)
            )
            j = torch.argmin(score)
            matched_idx[b] = j
            gt_xyz_match[b] = gt_traj_xyz[b, j]
            gt_t_match[b] = gt_ts[b, j]

        # ---- apply constrained spatial & temporal shifts ----
        delta = vlm_xyz - gt_xyz_match
        dist = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
        step = torch.clamp(dist, max=max_loc_shift_m)
        dirn = delta / dist
        shifted_xyz = gt_xyz_match + dirn * step  # spatially limited move

        dt = vlm_t - gt_t_match
        dt_clamped = dt.clamp(min=-max_time_shift, max=max_time_shift)
        shifted_t = gt_t_match + dt_clamped  # temporally limited move

        shifted_xyz_list.append(shifted_xyz)
        shifted_t_list.append(shifted_t)
        matched_idx_list.append(matched_idx)

    return shifted_xyz_list, shifted_t_list, matched_idx_list


class TrajConfig(PretrainedConfig):
    model_type = "traj"
    base_config_key = "traj_config"

    def __init__(
        self,
        input_dim=6,
        hidden_size=3584,
        attn_implementation_autoset=True,
        fm_n_head=4,
        fm_num_layers=6,
        action_dim=6,
        action_hidden_size=1024,
        time_max_period=100.0,
        fm_time_cond=True,
        head_dim=256,
        action_expert_rope_theta=100.0,
        fm_loss_alpha=5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self._attn_implementation_autoset = attn_implementation_autoset
        self.fm_n_head = fm_n_head
        self.fm_num_layers = fm_num_layers
        self.action_dim = action_dim
        self.fm_time_cond = fm_time_cond
        self.action_hidden_size = action_hidden_size
        self.time_max_period = time_max_period
        self.head_dim = head_dim
        self.fm_loss_alpha = fm_loss_alpha
        self.action_expert_rope_theta = action_expert_rope_theta


class EgoMANConfig(Qwen2_5_VLConfig):
    model_type = "egoman"
    base_config_key = "egoman_config"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.traj_config = TrajConfig()


# Traj block
class TrajBlock(nn.Module):
    def __init__(self, config: TrajConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(18, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, traj_seq):  # traj_seq: (B, T, 6)
        return self.layer_norm(self.proj(traj_seq))  # (B, T, hidden_size)


class TrajModel(PreTrainedModel):
    config_class = TrajConfig
    _no_split_modules = ["TrajBlock"]

    def __init__(self, config: TrajConfig):
        super().__init__(config)
        self.traj_block = TrajBlock(config)

    def forward(self, traj_input):  # (B, T, 6)
        return self.traj_block(traj_input)  # (B, T, hidden_size)


# model
class EgoMAN(Qwen2_5_VLForConditionalGeneration):
    config_class = EgoMANConfig

    def __init__(self, config):
        super().__init__(config)
        self.traj_encoder = TrajModel._from_config(config.traj_config)
        self.start_decoder = TrajDecoder(token_dim=self.config.hidden_size)
        self.contact_decoder = TrajDecoder(token_dim=self.config.hidden_size)
        self.end_decoder = TrajDecoder(token_dim=self.config.hidden_size)
        self.act_semantic_decoder = SemanticAligner(hidden_dim=self.config.hidden_size)
        self.fm_model = FMTransformerDecoder6DoF(
            cond_dim=768,
            action_dim=18,
            hidden_dim=768,
            num_encoder_layers=6,
            num_decoder_layers=6,
            nhead=8,
            time_emb_dim=256,
        )
        self.post_init()

    def pad_hidden(self, hiddens, target_len, D, device, dtype):
        if len(hiddens) > 0:
            stacked = torch.stack(hiddens, dim=0)  # [len_found, D]
        else:
            stacked = torch.zeros(0, D, device=device, dtype=dtype)
        if stacked.size(0) < target_len:
            pad_rows = target_len - stacked.size(0)
            padding = torch.zeros(pad_rows, D, device=device, dtype=dtype)
            stacked = torch.cat([stacked, padding], dim=0)
        elif stacked.size(0) > target_len:
            stacked = stacked[:target_len]
        return stacked

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pre_traj: Optional[torch.FloatTensor] = None,
        gt_traj: Optional[torch.FloatTensor] = None,
        future_valid: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        skip_get_nosie: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.


        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            pre_traj_positions = (input_ids == self.config.pre_traj_token_id).nonzero(
                as_tuple=False
            )
            if pre_traj is not None and pre_traj_positions.numel() > 0:
                pre_traj_embeds = self.traj_encoder(
                    pre_traj.to(inputs_embeds.device, inputs_embeds.dtype)
                ).to(inputs_embeds.device, inputs_embeds.dtype)
                batch_idx, seq_idx = pre_traj_positions[:, 0], pre_traj_positions[:, 1]
                inputs_embeds[batch_idx, seq_idx] = pre_traj_embeds.view(
                    -1, inputs_embeds.size(-1)
                )

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas

            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask.contiguous(),
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.contiguous(),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            B = len(inputs_embeds)
            act_pos = (shift_labels == self.config.act_token_id).nonzero(as_tuple=True)[
                0
            ]
            contact_pos = (shift_labels == self.config.contact_token_id).nonzero(
                as_tuple=True
            )[0]
            end_pos = (shift_labels == self.config.end_token_id).nonzero(as_tuple=True)[
                0
            ]
            start_pos = (shift_labels == self.config.start_token_id).nonzero(
                as_tuple=True
            )[0]

            act_hiddens = []
            start_hiddens = []
            contact_hiddens = []
            end_hiddens = []

            shift_hidden = hidden_states[0, :-1, :].contiguous()
            for pos in start_pos:
                start_hiddens.append(shift_hidden[pos])  # [D]

            for pos in contact_pos:
                contact_hiddens.append(shift_hidden[pos])  # [D]

            for pos in end_pos:
                end_hiddens.append(shift_hidden[pos])  # [D]

            for pos in act_pos:
                act_hiddens.append(shift_hidden[pos])

            start_hidden = self.pad_hidden(
                start_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            start_vec = gt_traj[:, 0]
            valid_rows = start_vec[(start_vec != -1000).any(dim=1)]  # [M, C]
            gt_start_vec = torch.full_like(start_vec, -1000.0)
            gt_start_vec[: valid_rows.size(0)] = valid_rows
            start_pred_time, start_pred_coords, start_total_loss = self.start_decoder(
                start_hidden, gt_start_vec
            )
            start_total_loss = start_total_loss * 0.3

            contact_hidden = self.pad_hidden(
                contact_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            contact_vec = gt_traj[:, 1]
            valid_rows = contact_vec[(contact_vec != -1000).any(dim=1)]  # [M, C]
            gt_contact_vec = torch.full_like(contact_vec, -1000.0)
            gt_contact_vec[: valid_rows.size(0)] = valid_rows
            contact_pred_time, contact_pred_coords, contact_total_loss = (
                self.contact_decoder(contact_hidden, gt_contact_vec)
            )
            contact_total_loss = contact_total_loss * 0.3

            end_hidden = self.pad_hidden(
                end_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            end_vec = gt_traj[:, 2]
            valid_rows = end_vec[(end_vec != -1000).any(dim=1)]  # [M, C]
            gt_end_vec = torch.full_like(end_vec, -1000.0)
            gt_end_vec[: valid_rows.size(0)] = valid_rows
            end_pred_time, end_pred_coords, end_total_loss = self.end_decoder(
                end_hidden, gt_end_vec
            )
            end_total_loss = end_total_loss * 0.3

            act_hidden = self.pad_hidden(
                act_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            act_vec = future_valid[:, 0][..., :768]
            valid_rows = act_vec[(act_vec != -1000).any(dim=1)]  # [M, C]
            gt_act_vec = torch.full_like(act_vec, -1000.0)
            gt_act_vec[: valid_rows.size(0)] = valid_rows

            act_loss, act_pred_emb = self.act_semantic_decoder(
                act_hidden, gt_act_vec.reshape(-1, 768)
            )
            act_loss = act_loss * 0.1

            waypoint_loss = (
                start_total_loss + contact_total_loss + end_total_loss + act_loss
            )
            with autocast(dtype=torch.float32):
                self.fm_model = self.fm_model.to(torch.float32)

                prev_traj = gt_traj[:, 3:8, 1:7]
                cur_gt_traj = gt_traj[:, 8:, 1:7]
                prev_rot = gt_traj[:, 3:8, 11:]
                cur_gt_rot = gt_traj[:, 8:, 11:]
                prev_traj = torch.cat(
                    [
                        prev_traj.reshape(len(gt_traj), -1, 2, 3),
                        prev_rot.reshape(len(gt_traj), -1, 2, 6),
                    ],
                    dim=-1,
                ).flatten(-2, -1)
                cur_gt_traj = torch.cat(
                    [
                        cur_gt_traj.reshape(len(gt_traj), -1, 2, 3),
                        cur_gt_rot.reshape(len(gt_traj), -1, 2, 6),
                    ],
                    dim=-1,
                ).flatten(-2, -1)
                valid_mask = torch.zeros(len(gt_traj), 60, dtype=torch.bool).to(
                    hidden_states.device
                )
                valid_mask[:, :10] = True  # one condition + 5 past + 3 waypoints
                valid_mask[:, 10:] = (cur_gt_traj != -1000).any(dim=-1)  # future

                context_embeds = future_valid[:, 0][..., 768:]
                context_embeds = self.fm_model.vis_proj(
                    context_embeds.to(torch.float32)
                ) + self.fm_model.text_proj(act_pred_emb.to(torch.float32))
                context_embeds = context_embeds.unsqueeze(1)
                valid_mask = valid_mask[..., 1:]

                left_gt_xyz_for_gate = cur_gt_traj[..., :3]
                right_gt_xyz_for_gate = cur_gt_traj[..., 9:12]
                B, T, _ = left_gt_xyz_for_gate.shape
                gt_time_grid = (
                    torch.arange(T, device=left_gt_xyz_for_gate.device)
                    .float()
                    .unsqueeze(0)
                    .repeat(B, 1)
                ).long()
                contact_pred_time = (
                    (contact_pred_time.clamp(min=0, max=1) * 4.5 * 10).round().long()
                )
                end_pred_time = (
                    (end_pred_time.clamp(min=0, max=1) * 4.5 * 10).round().long()
                )

                ### relax
                plaus_flags1 = plausibility_gate(
                    vlm_wpt_xyz=contact_pred_coords[..., :3],
                    vlm_t=contact_pred_time[..., 0],
                    gt_traj_xyz=left_gt_xyz_for_gate,
                    gt_ts=gt_time_grid,
                )[1]
                plaus_flags2 = plausibility_gate(
                    vlm_wpt_xyz=contact_pred_coords[..., 9:12],
                    vlm_t=contact_pred_time[..., 0],
                    gt_traj_xyz=right_gt_xyz_for_gate,
                    gt_ts=gt_time_grid,
                )[1]
                plaus_flags3 = plausibility_gate(
                    vlm_wpt_xyz=end_pred_coords[..., :3],
                    vlm_t=end_pred_time[..., 0],
                    gt_traj_xyz=left_gt_xyz_for_gate,
                    gt_ts=gt_time_grid,
                )[1]
                plaus_flags4 = plausibility_gate(
                    vlm_wpt_xyz=end_pred_coords[..., 9:12],
                    vlm_t=end_pred_time[..., 0],
                    gt_traj_xyz=right_gt_xyz_for_gate,
                    gt_ts=gt_time_grid,
                )[1]
                plausible_left = plaus_flags1 & plaus_flags3
                plausible_right = plaus_flags2 & plaus_flags4

                if plausible_left.sum() == 0:
                    vlm_xyz_list = [
                        contact_pred_coords[..., :3],  # contact left
                        end_pred_coords[..., :3],  # end left
                    ]
                    vlm_t_list = [
                        contact_pred_time[..., 0],
                        end_pred_time[..., 0],
                    ]
                    shifted_xyz_list, shifted_t_list, matched_idx_list = (
                        constrained_shift_xyz_time(
                            left_gt_xyz_for_gate, gt_time_grid, vlm_xyz_list, vlm_t_list
                        )
                    )
                    contact_pred_coords[..., :3] = shifted_xyz_list[0]
                    end_pred_coords[..., :3] = shifted_xyz_list[1]
                    contact_pred_time[..., 0] = shifted_t_list[0]
                    end_pred_time[..., 0] = shifted_t_list[1]
                if plausible_right.sum() == 0:
                    vlm_xyz_list = [
                        contact_pred_coords[..., 9:12],  # contact right
                        end_pred_coords[..., 9:12],  # end right
                    ]
                    vlm_t_list = [
                        contact_pred_time[..., 0],
                        end_pred_time[..., 0],
                    ]
                    shifted_xyz_list, shifted_t_list, matched_idx_list = (
                        constrained_shift_xyz_time(
                            right_gt_xyz_for_gate,
                            gt_time_grid,
                            vlm_xyz_list,
                            vlm_t_list,
                        )
                    )
                    contact_pred_coords[..., 9:12] = shifted_xyz_list[0]
                    end_pred_coords[..., 9:12] = shifted_xyz_list[1]
                    contact_pred_time[..., 0] = shifted_t_list[0]
                    end_pred_time[..., 0] = shifted_t_list[1]
                ### relax

                all_prev_traj = torch.cat(
                    [
                        prev_traj,
                        start_pred_coords.unsqueeze(1),
                        contact_pred_coords.unsqueeze(1),
                        end_pred_coords.unsqueeze(1),
                    ],
                    dim=1,
                )
                timestamp = torch.cat(
                    [start_pred_time * 0.0, contact_pred_time, end_pred_time], 1
                )
                timestamp = (timestamp).round().long().clamp(min=0, max=54)

                pred_v, target_v, sample_t, fm_loss = self.fm_model(
                    context_embeds.to(torch.float32),  # B, 768
                    all_prev_traj.to(torch.float32),
                    cur_gt_traj.to(torch.float32),
                    mask=1 - valid_mask.to(torch.float32),
                    loss=True,
                    timestamp=timestamp,
                )

            loss = loss + fm_loss

        if not return_dict:
            output = (logits,) + outputs[1:]

            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class EgoMAN_ReasonPretrain(EgoMAN):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pre_traj: Optional[torch.FloatTensor] = None,
        gt_traj: Optional[torch.FloatTensor] = None,
        future_valid: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        skip_get_nosie: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.


        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            pre_traj_positions = (input_ids == self.config.pre_traj_token_id).nonzero(
                as_tuple=False
            )
            if pre_traj is not None and pre_traj_positions.numel() > 0:
                pre_traj_embeds = self.traj_encoder(
                    pre_traj.to(inputs_embeds.device, inputs_embeds.dtype)
                ).to(inputs_embeds.device, inputs_embeds.dtype)
                batch_idx, seq_idx = pre_traj_positions[:, 0], pre_traj_positions[:, 1]
                inputs_embeds[batch_idx, seq_idx] = pre_traj_embeds.view(
                    -1, inputs_embeds.size(-1)
                )

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas

            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask.contiguous(),
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.contiguous(),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            B = len(inputs_embeds)
            act_pos = (shift_labels == self.config.act_token_id).nonzero(as_tuple=True)[
                0
            ]
            contact_pos = (shift_labels == self.config.contact_token_id).nonzero(
                as_tuple=True
            )[0]
            end_pos = (shift_labels == self.config.end_token_id).nonzero(as_tuple=True)[
                0
            ]
            start_pos = (shift_labels == self.config.start_token_id).nonzero(
                as_tuple=True
            )[0]

            act_hiddens = []
            start_hiddens = []
            contact_hiddens = []
            end_hiddens = []

            shift_hidden = hidden_states[0, :-1, :].contiguous()
            for pos in start_pos:
                start_hiddens.append(shift_hidden[pos])  # [D]

            for pos in contact_pos:
                contact_hiddens.append(shift_hidden[pos])  # [D]

            for pos in end_pos:
                end_hiddens.append(shift_hidden[pos])  # [D]

            for pos in act_pos:
                act_hiddens.append(shift_hidden[pos])

            start_hidden = self.pad_hidden(
                start_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            start_vec = gt_traj[:, 0]
            valid_rows = start_vec[(start_vec != -1000).any(dim=1)]  # [M, C]
            gt_start_vec = torch.full_like(start_vec, -1000.0)
            gt_start_vec[: valid_rows.size(0)] = valid_rows
            start_pred_time, start_pred_coords, start_total_loss = self.start_decoder(
                start_hidden, gt_start_vec
            )
            loss = loss + 0.3 * start_total_loss

            contact_hidden = self.pad_hidden(
                contact_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            contact_vec = gt_traj[:, 1]
            valid_rows = contact_vec[(contact_vec != -1000).any(dim=1)]  # [M, C]
            gt_contact_vec = torch.full_like(contact_vec, -1000.0)
            gt_contact_vec[: valid_rows.size(0)] = valid_rows
            contact_pred_time, contact_pred_coords, contact_total_loss = (
                self.contact_decoder(contact_hidden, gt_contact_vec)
            )
            loss = loss + 0.3 * contact_total_loss

            end_hidden = self.pad_hidden(
                end_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            end_vec = gt_traj[:, 2]
            valid_rows = end_vec[(end_vec != -1000).any(dim=1)]  # [M, C]
            gt_end_vec = torch.full_like(end_vec, -1000.0)
            gt_end_vec[: valid_rows.size(0)] = valid_rows
            end_pred_time, end_pred_coords, end_total_loss = self.end_decoder(
                end_hidden, gt_end_vec
            )
            loss = loss + 0.3 * end_total_loss

            act_hidden = self.pad_hidden(
                act_hiddens,
                len(gt_traj),
                shift_hidden.shape[-1],
                shift_hidden.device,
                shift_hidden.dtype,
            )
            act_vec = future_valid[:, 0]
            valid_rows = act_vec[(act_vec != -1000).any(dim=1)]  # [M, C]
            gt_act_vec = torch.full_like(act_vec, -1000.0)
            gt_act_vec[: valid_rows.size(0)] = valid_rows

            if dist.is_initialized():
                act_hidden = all_gather_tensor(act_hidden)
                gt_act_vec = all_gather_tensor(gt_act_vec)

            act_loss, act_pred_emb = self.act_semantic_decoder(
                act_hidden, gt_act_vec.reshape(-1, 768)
            )
            loss = loss + 0.1 * act_loss

        if not return_dict:
            output = (logits,) + outputs[1:]

            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


# infer
if __name__ == "__main__":
    egoman_model = EgoMAN.from_pretrained("../../data/weights/EgoMAN-7B")
