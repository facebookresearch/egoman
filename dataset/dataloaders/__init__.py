# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import re

data_root = "../data"
EgoMAN_Pretrain = {
    "annotation_path": f"{data_root}/egoman_dataset/egoman_pretrain.pkl",
    "data_path": f"{data_root}/egoman_imgs",
}
EgoMAN_Finetune = {
    "annotation_path": f"{data_root}/egoman_dataset/egoman_finetune.pkl",
    "data_path": f"{data_root}/egoman_imgs",
}
data_dict = {
    "egoman_pretrain": EgoMAN_Pretrain,
    "egoman_finetune": EgoMAN_Finetune,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["egoman_pretrain"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
