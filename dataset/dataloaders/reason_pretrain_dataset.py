# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import itertools
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from collections.abc import Sequence

import numpy as np
import torch
import transformers
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset, DistributedSampler
from torchcodec.decoders import VideoDecoder

from . import data_list
from .rope2d import get_rope_index_25

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    ori_sources: Optional[List] = None,
) -> Dict:

    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if (
        not hasattr(tokenizer, "chat_template")
        or tokenizer.chat_template != chat_template
    ):
        tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        # Handle case where grid_thw_video is None (video converted to image)
                        if (
                            grid_thw_video is not None
                            and visual_replicate_index_video < len(grid_thw_video)
                        ):
                            replacement = (
                                "<|vision_start|>"
                                + f"<|video_pad|>"
                                * grid_thw_video[visual_replicate_index_video][0]
                                + "<|vision_end|>"
                            )
                        else:
                            # Video was converted to image, treat as image
                            replacement = (
                                "<|vision_start|>"
                                + f"<|image_pad|>"
                                * grid_thw_image[visual_replicate_index_image]
                                + "<|vision_end|>"
                            )
                            visual_replicate_index_image += 1
                        new_parts.append(replacement)
                        if grid_thw_video is not None:
                            visual_replicate_index_video += 1
                    new_parts.append(parts[-1])

                content = "Given the history visual: " + "".join(new_parts)

                content = content.replace(
                    "Where should the left and right hands of the subject move if <action>?",
                    ori_sources[0]["question"],
                )

            if role == "assistant":

                content = ori_sources[0]["answer"]

            conv = [{"role": role, "content": content}]
            # print(conv)
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        self.get_rope_index = get_rope_index_25

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            elif file_format == "pkl":
                annotations = pickle.load(open(data["annotation_path"], "rb"))
            else:
                annotations = json.load(open(data["annotation_path"], "r"))

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
        self.data_args.image_processor.model_input_names.append("pre_traj")
        self.data_args.image_processor.model_input_names.append("gt_traj")
        self.data_args.image_processor.model_input_names.append("future_valid")
        self.invalid_list = []
        self.act_emb_dict = pickle.load(
            open(
                "../data/egoman_dataset/act_emb_dict.pkl",
                "rb",
            )
        )

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file, ratio=1.0, start_time=None, end_time=None):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 1
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(
                    video_file, ratio=ratio, start_time=start_time, end_time=end_time
                )
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file, ratio=ratio)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file, ratio=1.0, start_time=None, end_time=None):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            return None

        vr = VideoReader(video_file, num_threads=4)
        avg_fps = vr.get_avg_fps()
        total_frames = int(len(vr) * ratio)

        # Convert time (sec) to frame indices
        start_frame = (
            int(start_time * avg_fps)
            if start_time is not None
            else total_frames - int(avg_fps * 0.5)
        )
        end_frame = int(end_time * avg_fps) if end_time is not None else total_frames

        # Clamp the frame indices
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        # Compute interval and number of frames to sample
        interval = getattr(self.data_args, "base_interval", 4)
        video_length = (end_frame - start_frame) / avg_fps
        # print(avg_fps, start_frame, end_frame, video_length)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )

        # Sample frames between start_frame and end_frame
        frame_idx = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)

        # Load video frames
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file, ratio=1.0):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        total_frames = round(total_frames * ratio)
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 1
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                while i in self.invalid_list:
                    i = min(i + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                # print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(0.5)
                self.invalid_list.append(i)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_final_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                # print(
                #     f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                #     e,
                # )
                self.invalid_list.append(i)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = copy.deepcopy(self.list_data_dict[i])
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        mid_id = sources[0]["start_sec"]
        # Set up conversation - will be modified later if video is converted to image
        conversation_value = f"<image>\n Where should the left and right hands of the subject move if <action>?"
        sources[0]["conversations"] = [
            {
                "from": "human",
                "value": conversation_value,
            },
            {
                "from": "gpt",
                "value": "The predicted trajectory of the left and right hands of the subject are:\n<wrist>.",
            },
        ]

        future_act = torch.zeros(768) - 1000
        sources[0]["pre_wrist_pose"] = np.zeros((5, 18))
        sources[0]["wrist_pose"] = np.zeros((3, 23)) - 1000
        if "field" in sources[0]:
            if (
                len(sources[0]["field"]) == 5
                and sources[0]["answer"] == f"<ACT><START><CONTACT><END>"
            ):
                sources[0]["pre_wrist_pose"][..., :6] = np.array(sources[0]["value"][1])
                sources[0]["wrist_pose"][:3, :11] = np.array(sources[0]["value"][0])
                if "full_pos_vec" in sources[0]:
                    sources[0]["pre_wrist_pose"][..., :6] = np.array(
                        sources[0]["full_pos_vec"][:5, 1:7]
                    )
                    sources[0]["pre_wrist_pose"][..., 6:] = np.array(
                        sources[0]["full_pos_vec"][:5, 11:]
                    )
                    sources[0]["wrist_pose"] = np.array(sources[0]["full_pos_vec"][5:])
                else:
                    print(sources[0]["image"])

                # visible hoi
                if (
                    "left hand" in sources[0]["value"][2].lower()
                    and "right hand" not in sources[0]["value"][2].lower()
                ):
                    sources[0]["question"] = sources[0]["question"].replace(
                        sources[0]["intention"],
                        sources[0]["intention"] + " with left hand",
                    )
                    if (np.array(sources[0]["wrist_pose"][2][7:9]) == -1000).any():
                        sources[0]["wrist_pose"][2] = np.zeros(23) - 1000
                    if (np.array(sources[0]["wrist_pose"][1][7:9]) == -1000).any():
                        sources[0]["wrist_pose"][1] = np.zeros(23) - 1000
                elif (
                    "right hand" in sources[0]["value"][2].lower()
                    and "left hand" not in sources[0]["value"][2].lower()
                ):
                    sources[0]["question"] = sources[0]["question"].replace(
                        sources[0]["intention"],
                        sources[0]["intention"] + " with right hand",
                    )
                    if (np.array(sources[0]["wrist_pose"][2][9:11]) == -1000).any():
                        sources[0]["wrist_pose"][2] = np.zeros(23) - 1000
                    if (np.array(sources[0]["wrist_pose"][1][9:11]) == -1000).any():
                        sources[0]["wrist_pose"][1] = np.zeros(23) - 1000
                else:
                    sources[0]["question"] = sources[0]["question"].replace(
                        sources[0]["intention"],
                        sources[0]["intention"] + " with both hands",
                    )
                    left_contact = True
                    right_contact = True
                    left_end = True
                    right_end = True
                    if (np.array(sources[0]["wrist_pose"][2][7:9]) == -1000).any():
                        left_end = False
                    if (np.array(sources[0]["wrist_pose"][2][9:11]) == -1000).any():
                        right_end = False
                    if (np.array(sources[0]["wrist_pose"][1][7:9]) == -1000).any():
                        left_contact = False
                    if (np.array(sources[0]["wrist_pose"][1][9:11]) == -1000).any():
                        right_contact = False
                    if not left_contact:
                        sources[0]["wrist_pose"][1, 1:4] = -1000
                        sources[0]["wrist_pose"][1, 7:9] = -1000
                        sources[0]["wrist_pose"][1, 11:17] = -1000
                    if not right_contact:
                        sources[0]["wrist_pose"][1, 4:7] = -1000
                        sources[0]["wrist_pose"][1, 9:11] = -1000
                        sources[0]["wrist_pose"][1, 17:] = -1000
                    if not left_end:
                        sources[0]["wrist_pose"][2, 1:4] = -1000
                        sources[0]["wrist_pose"][2, 7:9] = -1000
                        sources[0]["wrist_pose"][2, 11:17] = -1000
                    if not right_end:
                        sources[0]["wrist_pose"][2, 4:7] = -1000
                        sources[0]["wrist_pose"][2, 9:11] = -1000
                        sources[0]["wrist_pose"][2, 17:] = -1000
                    if not left_contact and not right_contact:
                        sources[0]["wrist_pose"][1] = np.zeros(23) - 1000
                    if not left_end and not right_end:
                        sources[0]["wrist_pose"][2] = np.zeros(23) - 1000

                phrase_str = sources[0]["value"][2]
                key = sources[0]["image"] + "_" + phrase_str
                future_act = torch.Tensor(self.act_emb_dict[key]["emb"]).reshape(-1)

            else:
                sources[0]["question"] = sources[0]["question"] + "<HOI_QUERY>"
                # interaction q
                if "past_motion" in sources[0]["field"]:
                    sources[0]["pre_wrist_pose"][..., :6] = np.array(
                        sources[0]["value"][-1]
                    )
                    if "full_pos_vec" in sources[0]:
                        sources[0]["pre_wrist_pose"][..., :6] = np.array(
                            sources[0]["full_pos_vec"][:5, 1:7]
                        )
                        sources[0]["pre_wrist_pose"][..., 6:] = np.array(
                            sources[0]["full_pos_vec"][:5, 11:]
                        )
                    else:
                        print(sources[0]["image"])
                    all_fields = sources[0]["field"][:-1]
                else:
                    sources[0]["question"] = (
                        "Without the past wrist motion: <past_motion><past_motion><past_motion><past_motion><past_motion>. "
                        + sources[0]["question"]
                    )
                    all_fields = sources[0]["field"]

                for fid, cur_field in enumerate(all_fields):
                    if "start." in cur_field:
                        sources[0]["wrist_pose"][0, :11] = (
                            np.array(sources[0]["value"][fid]).reshape(-1).copy()
                        )
                        if "full_pos_vec" in sources[0]:
                            if ((sources[0]["wrist_pose"][0, 1:4]) == -1000).any():
                                sources[0]["full_pos_vec"][5, 1:4] = -1000
                                sources[0]["full_pos_vec"][5, 7:9] = -1000
                                sources[0]["full_pos_vec"][5, 11:17] = -1000
                            if ((sources[0]["wrist_pose"][0, 4:7]) == -1000).any():
                                sources[0]["full_pos_vec"][5, 4:7] = -1000
                                sources[0]["full_pos_vec"][5, 9:11] = -1000
                                sources[0]["full_pos_vec"][5, 17:23] = -1000
                            if sources[0]["wrist_pose"][0, 0] == -1000:
                                sources[0]["full_pos_vec"][5, 0] = -1000
                            sources[0]["wrist_pose"][0] = np.array(
                                sources[0]["full_pos_vec"][5]
                            )
                        else:
                            print(sources[0]["image"])

                    elif "end." in cur_field:
                        sources[0]["wrist_pose"][2, :11] = (
                            np.array(sources[0]["value"][fid]).reshape(-1).copy()
                        )
                        if "full_pos_vec" in sources[0]:
                            if ((sources[0]["wrist_pose"][2, 1:4]) == -1000).any():
                                sources[0]["full_pos_vec"][7, 1:4] = -1000
                                sources[0]["full_pos_vec"][7, 7:9] = -1000
                                sources[0]["full_pos_vec"][7, 11:17] = -1000
                            if ((sources[0]["wrist_pose"][2, 4:7]) == -1000).any():
                                sources[0]["full_pos_vec"][7, 4:7] = -1000
                                sources[0]["full_pos_vec"][7, 9:11] = -1000
                                sources[0]["full_pos_vec"][7, 17:23] = -1000
                            if sources[0]["wrist_pose"][2, 0] == -1000:
                                sources[0]["full_pos_vec"][7, 0] = -1000
                            sources[0]["wrist_pose"][2] = np.array(
                                sources[0]["full_pos_vec"][7]
                            )
                        else:
                            print(sources[0]["image"])
                    elif "contact." in cur_field:
                        sources[0]["wrist_pose"][1, :11] = (
                            np.array(sources[0]["value"][fid]).reshape(-1).copy()
                        )
                        if "full_pos_vec" in sources[0]:
                            if ((sources[0]["wrist_pose"][1, 1:4]) == -1000).any():
                                sources[0]["full_pos_vec"][6, 1:4] = -1000
                                sources[0]["full_pos_vec"][6, 7:9] = -1000
                                sources[0]["full_pos_vec"][6, 11:17] = -1000
                            if ((sources[0]["wrist_pose"][1, 4:7]) == -1000).any():
                                sources[0]["full_pos_vec"][6, 4:7] = -1000
                                sources[0]["full_pos_vec"][6, 9:11] = -1000
                                sources[0]["full_pos_vec"][6, 17:23] = -1000
                            if sources[0]["wrist_pose"][1, 0] == -1000:
                                sources[0]["full_pos_vec"][6, 0] = -1000
                            sources[0]["wrist_pose"][1] = np.array(
                                sources[0]["full_pos_vec"][6]
                            )
                        else:
                            print(sources[0]["image"])
                    elif cur_field == "act.phrase":
                        phrase_str = sources[0]["value"][fid]
                        key = sources[0]["image"] + "_" + phrase_str
                        future_act = torch.Tensor(
                            self.act_emb_dict[key]["emb"]
                        ).reshape(-1)
        else:
            sources[0]["question"] = (
                "Without the past wrist motion: <past_motion><past_motion><past_motion><past_motion><past_motion>. "
                + sources[0]["question"]
            )
        sources[0].pop("video")
        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None
        sources[0]["wrist_pose"][0, 0] = -1000

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]

        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        # print(video_grid_thw)
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
            ori_sources=sources,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw=grid_thw_merged,
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        pre_traj = torch.Tensor(np.array(sources[0]["pre_wrist_pose"]))
        data_dict["pre_traj"] = pre_traj
        gt_traj = torch.Tensor(np.array(sources[0]["wrist_pose"]))
        data_dict["gt_traj"] = gt_traj

        data_dict["future_valid"] = future_act.reshape(1, -1)

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        pre_traj_list = list(
            instance["pre_traj"] for instance in instances if "pre_traj" in instance
        )
        traj_list = list(
            instance["gt_traj"] for instance in instances if "gt_traj" in instance
        )
        future_valid_list = list(
            instance["future_valid"]
            for instance in instances
            if "future_valid" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if len(pre_traj_list) != 0:
            concat_pre_traj = torch.cat(
                [pre_traj for pre_traj in pre_traj_list], dim=0
            ).unsqueeze(0)
        else:
            concat_pre_traj = None

        if len(traj_list) != 0:
            concat_traj = torch.stack([traj for traj in traj_list], dim=0)
        else:
            concat_traj = None

        if len(future_valid_list) != 0:
            concat_future_valid = torch.stack(
                [future_valid for future_valid in future_valid_list], dim=0
            )
        else:
            concat_future_valid = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        batch["pre_traj"] = concat_pre_traj
        batch["gt_traj"] = concat_traj
        batch["future_valid"] = concat_future_valid
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        pre_traj_list = list(
            instance["pre_traj"] for instance in instances if "pre_traj" in instance
        )
        traj_list = list(
            instance["gt_traj"] for instance in instances if "gt_traj" in instance
        )
        future_valid_list = list(
            instance["future_valid"]
            for instance in instances
            if "future_valid" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if len(pre_traj_list) != 0:
            concat_pre_traj = torch.cat(
                [pre_traj for pre_traj in pre_traj_list], dim=0
            ).unsqueeze(0)
        else:
            concat_pre_traj = None

        if len(traj_list) != 0:
            concat_traj = torch.stack([traj for traj in traj_list], dim=0)
        else:
            concat_traj = None

        if len(future_valid_list) != 0:
            concat_future_valid = torch.stack(
                [future_valid for future_valid in future_valid_list], dim=0
            )
        else:
            concat_future_valid = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["pre_traj"] = concat_pre_traj
        batch["gt_traj"] = concat_traj
        batch["future_valid"] = concat_future_valid
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
