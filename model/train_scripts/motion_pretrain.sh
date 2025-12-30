# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


CUDA_VISIBLE_DEVICES=0 torchrun --master-port=16476 --nproc-per-node=1 tools/motion_pretrain.py
