# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import functools

import math
import os
import pickle
import random

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union
from collections.abc import Sequence
from datetime import timedelta

import _init_paths

import numpy as np
import pytorch3d.transforms as pt

import torch
import torch.distributed as dist
import transformers
from egoman_model import EgoMAN
from PIL import Image

from PIL.Image import new

from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.feature_extraction_utils import BatchFeature


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=20)
    video_min_frames: Optional[int] = field(default=1)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=0.25)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=1 * 28 * 28)
    max_scale: float = field(default=1.0)
    z_neg: bool = field(default=False)


def _now():
    return time.perf_counter()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


model_name = os.environ.get("MODEL_NAME", "EgoMAN-7B")
video_id = "examples"
img_dir = f"../data/{video_id}"
vis_emb_dict = pickle.load(
    open(
        f"../data/{video_id}_vis_features.pkl",
        "rb",
    )
)
CONFIG = {
    "batch_size": int(os.environ.get("BATCH_SIZE", 8)),
    "save_every": int(os.environ.get("SAVE_EVERY", 200)),
    "model_path": f"../data/weights/{model_name}",
    "log_path": f"../output/{model_name.replace('/', '-')}-{video_id}.pkl",
    "seed": int(os.environ.get("SEED", 42)),
}
os.makedirs(os.path.dirname(CONFIG["log_path"]), exist_ok=True)


# ----------------------------
# DIST INIT/CLEANUP
# ----------------------------
def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=1),  # <- use datetime.timedelta
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_dist(is_dist):
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------
# HELPERS
# ----------------------------
def shard_indices(n_items, world_size, rank):
    return list(range(rank, n_items, world_size))


def chunk_list(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

provider = None


def process_image_unified(image_file):
    processor = copy.deepcopy(DataArguments.image_processor)

    processor.max_pixels = 50176
    processor.min_pixels = 784
    processor.size["longest_edge"] = processor.max_pixels
    processor.size["shortest_edge"] = processor.min_pixels

    image = Image.open(image_file).convert("RGB")

    visual_processed = processor.preprocess(image, return_tensors="pt")
    image_tensor = visual_processed["pixel_values"]
    if isinstance(image_tensor, List):
        image_tensor = image_tensor[0]
    grid_thw = visual_processed["image_grid_thw"][0]
    return image_tensor, grid_thw


def preprocess_visual(
    sources: List[List[Dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: Optional[Sequence[int]] = None,
    ori_sources: Optional[List[Dict]] = None,
    system_message: str = "You are a helpful assistant.",
    visual_prefix: str = "Given the history visual: ",
) -> Dict[str, torch.Tensor]:
    """
    Build Qwen2.5-VL chat prompts with visual placeholders.

    Args
    ----
    sources: batch of conversations; each item is a list of turns like:
        {"from":"human","value":"<image> ..."} or {"role":"user","content":"..."}
    tokenizer: HF tokenizer
    grid_thw_image: per-<image> placeholder counts AFTER merging; length must equal
        total number of <image> markers across the batch. Each entry is an int N
        and we emit N occurrences of <|image_pad|>.
    ori_sources: per-sample auxiliary dicts; used here to replace the template question
        with ori_sources[i]["question"].

    Returns
    -------
    {
      "input_ids": LongTensor [B, L_max],
      "input_lens": LongTensor [B],   # lengths before padding (optional convenience)
    }
    """
    roles_map = {"human": "user", "gpt": "assistant"}
    # ensure template is set (do this once outside if you prefer)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

    grid_thw_image = list(grid_thw_image) if grid_thw_image is not None else []

    img_idx = 0

    batch_input_ids = []
    input_lens = []

    for bi, source in enumerate(sources):
        # normalize turns: prefer keys ("role","content"); fallback to ("from","value")
        turns = []
        for t in source:
            if "role" in t and "content" in t:
                role = t["role"]
                content = t["content"]
            else:
                role = t.get("from", "")
                content = t.get("value", "")
            role = roles_map.get(role, role)
            turns.append({"role": role, "content": content})

        # If the first turn is not 'user' (e.g., a stray assistant), drop leading non-user
        if len(turns) > 0 and turns[0]["role"] != "user":
            # mirror original behavior: skip the first non-user entry
            turns = turns[1:]

        # Build a single message list: system + user turns only (skip assistant)
        messages = [{"role": "system", "content": system_message}]

        for t in turns:
            if t["role"] == "assistant":
                # we do not include assistant messages in inputs for generation
                continue

            content = t["content"]

            # Replace template question with sample-specific one
            if ori_sources is not None and bi < len(ori_sources):
                # only replace this exact template if present
                content = content.replace(
                    "Where should the left and right hands of the subject move if <action>?",
                    ori_sources[bi].get("question", ""),
                )

            # Expand <image> markers
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for _ in range(len(parts) - 1):
                    new_parts.append(parts.pop(0))
                    if img_idx >= len(grid_thw_image):
                        raise ValueError(
                            f"Not enough grid_thw_image entries: needed more at index {img_idx}."
                        )
                    n_patches = int(grid_thw_image[img_idx])
                    img_idx += 1
                    replacement = (
                        "<|vision_start|>"
                        + ("<|image_pad|>" * n_patches)
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[0])  # last remainder
                content = visual_prefix + "".join(new_parts)

            messages.append({"role": "user", "content": content})

        # encode once per sample with add_generation_prompt=True
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(
            0
        )  # (L,)
        batch_input_ids.append(ids)
        input_lens.append(ids.size(0))

    # final validation: all markers consumed?
    if img_idx != len(grid_thw_image):
        raise ValueError(
            f"Unused grid_thw_image entries: consumed {img_idx}, provided {len(grid_thw_image)}."
        )

    # pad to a single tensor [B, L_max]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        # Qwen typically has a pad token; if not, fall back to eos
        pad_id = tokenizer.eos_token_id
    max_len = max(input_lens)
    padded = [
        (
            torch.cat([ids, ids.new_full((max_len - ids.size(0),), pad_id)])
            if ids.size(0) < max_len
            else ids
        )
        for ids in batch_input_ids
    ]
    input_ids = torch.stack(padded, dim=0)  # [B, L_max]

    return {
        "input_ids": input_ids.long(),
        "input_lens": torch.tensor(input_lens, dtype=torch.long),
    }


class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [
                getattr(pt, f"{from_rep}_to_matrix"),
                getattr(pt, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(func, convention=from_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                getattr(pt, f"matrix_to_{to_rep}"),
                getattr(pt, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [
                    functools.partial(func, convention=to_convention) for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: Union[np.ndarray, torch.Tensor], funcs: list
    ) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


rot_trans = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")
rotation_transformer = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")


# ----------------------------
# MAIN
# ----------------------------
def main():
    is_dist, rank, world_size, local_rank = setup_dist()

    # seeds
    random.seed(CONFIG["seed"] + rank)
    np.random.seed(CONFIG["seed"] + rank)
    torch.manual_seed(CONFIG["seed"] + rank)

    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    annotations_all = []
    img_all = [imgpath for imgpath in os.listdir(img_dir) if imgpath.endswith(".jpg")]
    for cur_img in img_all:
        cur_anno = {}
        quest = cur_img[:-4].replace("_", " ")
        cur_anno["intention"] = quest
        cur_anno["question"] = (
            f"Given the past wrist motion: <past_motion><past_motion><past_motion><past_motion><past_motion>. Where will the hands move to {cur_anno['intention']}?<HOI_QUERY>"
        )
        cur_anno["image"] = cur_img
        annotations_all.append(cur_anno.copy())

    indices = shard_indices(len(annotations_all), world_size, rank)

    # model/tokenizer/processor (per rank)
    _model_path = CONFIG["model_path"]
    model = EgoMAN.from_pretrained(_model_path, device_map=None)
    tokenizer = AutoTokenizer.from_pretrained(
        _model_path, repo_type="local", padding_side="right", use_fast=False
    )
    processor = AutoProcessor.from_pretrained(_model_path)
    processor.tokenizer = tokenizer
    DataArguments.image_processor = processor.image_processor

    model.eval().to(device)
    torch.cuda.empty_cache()

    local_results = []
    save_every = max(CONFIG["save_every"] // max(world_size, 1), 1)

    # batched loop per-rank
    bs = CONFIG["batch_size"]
    num_batches = math.ceil(len(indices) / bs)

    pbar = tqdm(
        total=num_batches,
        desc=f"rank {rank} batches",
        disable=(rank != 0),
        dynamic_ncols=True,
    )

    for batch_indices in chunk_list(indices, bs):
        pbar.set_postfix_str(f"{batch_indices[0]}–{batch_indices[-1]}")
        batch_ann = [annotations_all[i] for i in batch_indices]
        chat_sources, images, grid_thw_list, data_dicts = [], [], [], []

        for ori_cur_anno in batch_ann:
            image_file = f"{img_dir}/{ori_cur_anno['image']}"

            cur_anno = ori_cur_anno.copy()
            # ---- your prompt & QA prep (unchanged) ----
            cur_anno["conversations"] = [
                {
                    "from": "human",
                    "value": "<image>\n Where should the left and right hands of the subject move if <action>?",
                },
                {
                    "from": "gpt",
                    "value": "The predicted trajectory of the left and right hands of the subject are:\n<wrist>.",
                },
            ]

            if "answer" in ori_cur_anno:
                cur_anno["answer"] = ori_cur_anno["answer"]

            future_act = torch.zeros(768) - 1000
            cur_anno["pre_wrist_pose"] = np.zeros((5, 18))
            cur_anno["wrist_pose"] = np.zeros((3, 23)) - 1000

            past_motion = np.load(
                os.path.join(img_dir, ori_cur_anno["image"][:-4] + "+past_motion.npy")
            )
            quat_ori = torch.from_numpy(past_motion[..., 3:]).reshape(-1, 4)
            quat_ori = torch.stack(
                [quat_ori[:, 3], quat_ori[:, 0], quat_ori[:, 1], quat_ori[:, 2]],
                dim=1,
            )
            quat_ori = rot_trans.forward(quat_ori).reshape(-1, 12)

            cur_anno["pre_wrist_pose"][..., :6] = np.array(
                past_motion[..., :3]
            ).reshape(-1, 6)
            cur_anno["pre_wrist_pose"][..., 6:] = quat_ori[:5].detach().cpu().numpy()

            vis_key = ori_cur_anno["image"]
            dino_vis_emb = torch.from_numpy(vis_emb_dict[vis_key]).reshape(-1)
            future_act = torch.zeros(768) - 1000
            future_act = torch.cat([future_act, dino_vis_emb], dim=-1)

            try:
                img_tensor, grid_thw = process_image_unified(image_file)
            except:
                print(f"Failed to load image {image_file}")
                continue

            images.append(img_tensor)
            grid_thw_list.append(grid_thw)
            chat_sources.append(copy.deepcopy(cur_anno["conversations"]))

            pre_traj = torch.from_numpy(np.array(cur_anno["pre_wrist_pose"]))
            real_gt_traj = torch.zeros(53, 23) - 1000
            data_dicts.append(
                {
                    "pre_traj": pre_traj,
                    "gt_traj": real_gt_traj,
                    "future_valid": future_act,
                    "ori": cur_anno,
                }
            )

        if len(images) == 0:
            continue
        # build batched prompt with <image> paddings
        grid_thw_merged = [
            (
                (g.prod() // DataArguments.image_processor.merge_size**2)
                if hasattr(g, "prod")
                else int(np.prod(g) // (DataArguments.image_processor.merge_size**2))
            )
            for g in grid_thw_list
        ]
        enc = preprocess_visual(
            chat_sources,
            tokenizer,
            grid_thw_image=grid_thw_merged,
            ori_sources=[d["ori"] for d in data_dicts],
        )
        # enc provides already-padded [B, L_max] and per-sample lengths
        input_ids = enc["input_ids"].to(device)  # [B, L_max]
        input_lens = enc["input_lens"].tolist()  # [B]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        attention_mask = (input_ids != pad_id).long()

        # visuals
        pixel_values = torch.cat(images, dim=0).to(device)  # [B, C, H, W]
        image_grid_thw = torch.cat([g.unsqueeze(0) for g in grid_thw_list], dim=0).to(
            device
        )

        # pack batch
        inputs = BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "pre_traj": torch.stack([d["pre_traj"] for d in data_dicts], dim=0).to(
                    device
                ),
                "gt_traj": torch.stack([d["gt_traj"] for d in data_dicts], dim=0).to(
                    device
                ),
                "future_valid": torch.stack(
                    [d["future_valid"] for d in data_dicts], dim=0
                )
                .unsqueeze(1)
                .to(device),
            },
            tensor_type="pt",
        )
        # --- generate (batched) ---
        model.config.output_hidden_states = True
        t0 = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            torch.cuda.synchronize(device)
            # 1) trim each sample with its own prompt length
            generated_ids = outputs.sequences  # list/ tensor len B
            generated_ids_trimmed = [
                seq[L_i:] for seq, L_i in zip(generated_ids, input_lens)
            ]

            output_text_batch = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # 2) align hidden states per sample starting at (L_i - 1)
            last_hidden_states = [
                each[-1] for each in outputs.hidden_states
            ]  # list over layers
            all_hidden_states = torch.cat(last_hidden_states, dim=1)  # [B, T, H]
            per_sample_hiddens = [
                all_hidden_states[b, (L_i - 1) :, :].contiguous()
                for b, L_i in enumerate(input_lens)
            ]

            # ids for tags
            start_token_id = tokenizer.convert_tokens_to_ids("<START>")
            contact_token_id = tokenizer.convert_tokens_to_ids("<CONTACT>")
            end_token_id = tokenizer.convert_tokens_to_ids("<END>")
            act_token_id = tokenizer.convert_tokens_to_ids("<ACT>")

            for bx, (pred_txt, gen_ids, hids, dd) in enumerate(
                zip(
                    output_text_batch,
                    generated_ids_trimmed,
                    per_sample_hiddens,
                    data_dicts,
                )
            ):
                result = dd["ori"].copy()
                result["prediction"] = pred_txt.strip()

                gen_ids_cpu = gen_ids.to("cpu")
                start_mask = gen_ids_cpu == start_token_id
                contact_mask = gen_ids_cpu == contact_token_id
                end_mask = gen_ids_cpu == end_token_id
                act_mask = gen_ids_cpu == act_token_id

                if start_mask.any():
                    start_out = (
                        model.start_decoder.inference_stage2(hids[start_mask, :])[0]
                        .cpu()
                        .numpy()
                    )
                    result["pred_start"] = start_out.tolist()
                if contact_mask.any():
                    contact_out = (
                        model.contact_decoder.inference_stage2(hids[contact_mask, :])[0]
                        .cpu()
                        .numpy()
                    )
                    result["pred_contact"] = contact_out.tolist()
                if end_mask.any():
                    end_out = (
                        model.end_decoder.inference_stage2(hids[end_mask, :])[0]
                        .cpu()
                        .numpy()
                    )
                    result["pred_end"] = end_out.tolist()
                if act_mask.any():
                    act_out_emb = (
                        model.act_semantic_decoder.inference(hids[act_mask, :])[0]
                        .cpu()
                        .numpy()
                    )
                    result["pred_act_emb"] = act_out_emb.tolist()

                model.fm_model = model.fm_model.to(torch.float32)

                # FM Model trajectory prediction and metrics calculation
                if (
                    start_mask.any()
                    and contact_mask.any()
                    and end_mask.any()
                    and act_mask.any()
                ):
                    start_out_tensor = torch.from_numpy(start_out).to(device)
                    contact_out_tensor = torch.from_numpy(contact_out).to(device)
                    end_out_tensor = torch.from_numpy(end_out).to(device)
                    act_out_emb_tensor = torch.from_numpy(act_out_emb).to(device)

                    start_pred_time = start_out_tensor[..., 0].reshape(-1)
                    contact_pred_time = contact_out_tensor[..., 0].reshape(-1)
                    end_pred_time = end_out_tensor[..., 0].reshape(-1)

                    start_out_tensor = torch.cat(
                        [
                            start_out_tensor[..., 1:7].reshape(-1, 2, 3),
                            start_out_tensor[..., 7:].reshape(-1, 2, 6),
                        ],
                        dim=-1,
                    ).reshape(-1, 18)

                    contact_out_tensor = torch.cat(
                        [
                            contact_out_tensor[..., 1:7].reshape(-1, 2, 3),
                            contact_out_tensor[..., 7:].reshape(-1, 2, 6),
                        ],
                        dim=-1,
                    ).reshape(-1, 18)

                    end_out_tensor = torch.cat(
                        [
                            end_out_tensor[..., 1:7].reshape(-1, 2, 3),
                            end_out_tensor[..., 7:].reshape(-1, 2, 6),
                        ],
                        dim=-1,
                    ).reshape(-1, 18)

                    prev_traj = (
                        torch.cat(
                            [
                                dd["pre_traj"][:, :6].reshape(-1, 2, 3),
                                dd["pre_traj"][:, 6:].reshape(-1, 2, 6),
                            ],
                            dim=-1,
                        )
                        .flatten(-2, -1)
                        .unsqueeze(0)
                        .to(device)
                    )

                    valid_mask = torch.zeros(1, 60, dtype=torch.bool).to(device)
                    valid_mask[:, :10] = True
                    valid_mask[:, 10:] = True
                    valid_mask = valid_mask[..., 1:]

                    all_prev_traj = torch.cat(
                        [
                            prev_traj,
                            start_out_tensor.unsqueeze(1),
                            contact_out_tensor.unsqueeze(1),
                            end_out_tensor.unsqueeze(1),
                        ],
                        dim=1,
                    )

                    timestamp = torch.cat(
                        [
                            start_pred_time.unsqueeze(-1) * 0.0,
                            contact_pred_time.unsqueeze(-1),
                            end_pred_time.unsqueeze(-1),
                        ],
                        -1,
                    )
                    timestamp = (
                        (timestamp.clamp(min=0, max=1) * 4.5 * 10).round().long()
                    )

                    context_embeds = dd["future_valid"][768:].unsqueeze(0).to(device)
                    context_embeds = model.fm_model.vis_proj(
                        context_embeds.to(torch.float32)
                    ) + model.fm_model.text_proj(act_out_emb_tensor.to(torch.float32))
                    context_embeds = context_embeds.unsqueeze(1)
                    K = 10

                    pred_hands = model.fm_model.sample_trajectory(
                        hs_t=context_embeds.to(torch.float32).repeat(K, 1, 1),
                        prev_traj=all_prev_traj.to(torch.float32).repeat(K, 1, 1),
                        L_future=50,
                        action_dim=18,
                        device=device,
                        n_steps=150,
                        mask=1 - valid_mask.to(torch.float32).repeat(K, 1),
                        timestamp=timestamp.repeat(K, 1),
                    )

                    result["ori_pred_wrist_pose"] = pred_hands.data.cpu().numpy()
                    local_results.append(result.copy())

        pbar.update(1)
    pbar.close()

    # flush remaining - append to existing results
    tmp_path = CONFIG["log_path"].replace(".pkl", f".rank{rank}.tmp.pkl")
    if len(local_results) > 0:
        # Load existing results if file exists, then append
        existing_results = []
        if os.path.exists(tmp_path):
            existing_results = pickle.load(open(tmp_path, "rb"))
        existing_results.extend(local_results)
        pickle.dump(existing_results, open(tmp_path, "wb"))
        local_results.clear()

    # gather
    if is_dist:
        dist.barrier()

    if rank == 0:
        merged = []
        for r in range(world_size):
            rp = CONFIG["log_path"].replace(".pkl", f".rank{r}.tmp.pkl")
            if os.path.exists(rp):
                merged.extend(pickle.load(open(rp, "rb")))
                os.remove(rp)
        os.makedirs(os.path.dirname(CONFIG["log_path"]), exist_ok=True)
        pickle.dump(merged, open(CONFIG["log_path"], "wb"))
        print(f"[rank 0] wrote {len(merged)} results → {CONFIG['log_path']}")

    cleanup_dist(is_dist)


if __name__ == "__main__":
    main()

# BATCH_SIZE=1 SAVE_EVERY=50 MODEL_NAME=EgoMAN-7B torchrun --nproc_per_node=1 tools/infer_batch_aria_examples.py
