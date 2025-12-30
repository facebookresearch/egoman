# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import copy
import functools
import os
import pickle
import random
from typing import Union

import numpy as np
import pytorch3d.transforms as pt
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


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


class EgoMANDataset(Dataset):
    """Dataset for egocentric trajectory prediction with video and language integration"""

    def __init__(
        self,
        data_path="../data/egoman_dataset",
        split="train",
    ):
        super().__init__()
        if split == "train":
            self.data = pickle.load(
                open(
                    os.path.join(
                        data_path,
                        "egoman_finetune.pkl",
                    ),
                    "rb",
                )
            )
            self.text_feat = pickle.load(
                open(os.path.join(data_path, "act_emb_dict.pkl"), "rb")
            )
            print("train data size: ", len(self.data))
        else:
            self.data = pickle.load(
                open(
                    os.path.join(
                        data_path,
                        "egoman-test-final.pkl",
                    ),
                    "rb",
                )
            )
            self.text_feat = pickle.load(
                open(
                    os.path.join(data_path, "act_emb_val_dict.pkl"),
                    "rb",
                )
            )
        self.split = split
        self.img_feat = pickle.load(
            open(os.path.join(data_path, "egoman_dinov3_features.pkl"), "rb")
        )
        self.tf = RotationTransformer(
            from_rep="quaternion", to_rep="rotation_6d"
        )  # quaternion xyzw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx])
        action_text = sample["value"][2]
        intention_text = sample["intention"]
        image_path = sample["image"]
        text_key = f"{image_path}_{action_text}"
        text_feature = torch.Tensor(self.text_feat[text_key]["emb"]).unsqueeze(0)
        video_tensor = torch.from_numpy(self.img_feat[image_path]).unsqueeze(0)
        past_traj = torch.from_numpy(sample["value"][-2][:5])
        past_quat = torch.from_numpy(sample["value"][-1][:5])
        past_quat = past_quat.reshape(-1, 4)
        # xyzw to wxyz
        past_quat = torch.stack(
            [past_quat[:, 3], past_quat[:, 0], past_quat[:, 1], past_quat[:, 2]], dim=1
        )
        past_quat = self.tf.forward(past_quat).reshape(-1, 12)

        fut_traj = torch.zeros(50, 6)
        fut_quat = torch.zeros(50, 12)
        fut_traj_ori = torch.from_numpy(sample["value"][3])[5:]
        fut_traj[: len(fut_traj_ori)] = fut_traj_ori
        fut_quat_ori = torch.from_numpy(sample["value"][4])[5:]
        fut_quat_ori = fut_quat_ori.reshape(-1, 4)
        # xyzw to wxyz
        fut_quat_ori = torch.stack(
            [
                fut_quat_ori[:, 3],
                fut_quat_ori[:, 0],
                fut_quat_ori[:, 1],
                fut_quat_ori[:, 2],
            ],
            dim=1,
        )
        fut_quat_ori = self.tf.forward(fut_quat_ori).reshape(-1, 12)
        fut_quat[: len(fut_quat_ori)] = fut_quat_ori
        start_contact_loc = torch.from_numpy(sample["value"][0][..., 1:7]).reshape(
            -1, 2, 3
        )
        start_end_timestamp = (
            torch.from_numpy(sample["value"][0][..., 0] * 4.5 * 10).round().long()
        )
        time_idx = start_end_timestamp[1]
        start_contact_loc_quat = torch.stack(
            [fut_quat_ori[0], fut_quat_ori[time_idx], fut_quat_ori[-1]], dim=0
        ).reshape(-1, 2, 6)
        start_contact_loc = torch.cat(
            [start_contact_loc, start_contact_loc_quat], dim=-1
        )
        start_contact_loc = start_contact_loc.reshape(-1, 18)

        future_valid = torch.zeros(59, dtype=torch.bool)
        future_valid[:9] = True
        future_valid[9 : 9 + len(fut_traj_ori)] = True
        return {
            "past_traj": past_traj.reshape(-1, 2, 3).permute(1, 0, 2),
            "fut_traj": fut_traj.reshape(-1, 2, 3).permute(1, 0, 2),
            "past_quat": past_quat.reshape(-1, 2, 6).permute(1, 0, 2),
            "fut_quat": fut_quat.reshape(-1, 2, 6).permute(1, 0, 2),
            "start_contact_loc": start_contact_loc,
            "pixel_values_videos": video_tensor,
            "future_valid": future_valid,
            "action_text": action_text,
            "image_path": image_path,
            "intention_text": intention_text,
            "start_end_timestamp": start_end_timestamp,
            "text_feature": text_feature,
        }


def seq_collate_egoman(batch):
    """Collate function for egocentric trajectory data"""
    # Extract individual components based on getitem return keys
    past_traj = torch.stack([item["past_traj"] for item in batch])
    fut_traj = torch.stack([item["fut_traj"] for item in batch])
    start_contact_loc = torch.stack([item["start_contact_loc"] for item in batch])
    pixel_values_videos = torch.stack([item["pixel_values_videos"] for item in batch])
    future_valid = torch.stack([item["future_valid"] for item in batch])
    action_text = [item["action_text"] for item in batch]
    image_path = [item["image_path"] for item in batch]
    intention_text = [item["intention_text"] for item in batch]
    start_end_timestamp = torch.stack([item["start_end_timestamp"] for item in batch])
    text_feature = torch.stack([item["text_feature"] for item in batch])
    past_quat = torch.stack([item["past_quat"] for item in batch])
    fut_quat = torch.stack([item["fut_quat"] for item in batch])

    batch_size = len(batch)

    data = {
        "past_traj": past_traj,
        "fut_traj": fut_traj,
        "past_quat": past_quat,
        "fut_quat": fut_quat,
        "start_contact_loc": start_contact_loc,
        "pixel_values_videos": pixel_values_videos,
        "future_valid": future_valid,
        "action_text": action_text,
        "image_path": image_path,
        "intention_text": intention_text,
        "start_end_timestamp": start_end_timestamp,
        "text_feature": text_feature,
        "batch_size": torch.tensor(batch_size),
    }

    return data


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoTokenizer

    dataset = EgoMANDataset(split="train")
    import random

    random.shuffle(dataset.data)
    data = dataset.__getitem__(0)
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        collate_fn=seq_collate_egoman,
        num_workers=0,
    )
    for data in tqdm(train_dataloader):
        print(data.keys())
        break

    val_dataset = EgoMANDataset(split="val")
    random.shuffle(val_dataset.data)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=seq_collate_egoman,
        num_workers=0,
    )

    for data in tqdm(val_dataloader):
        print(data.keys())
        break
