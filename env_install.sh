# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 xformers torchcodec --index-url https://download.pytorch.org/whl/cu124
conda install ffmpeg -c conda-forge -y
pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124
pip install imageio
pip install opencv-python
pip install decord
pip install qwen_vl_utils
pip install pandas

pip install triton==3.0.0 accelerate==1.4.0 --no-deps
pip install huggingface_hub
pip install safetensors
pip install datasets
pip install tokenizers
pip install py-cpuinfo
pip install deepspeed==0.16.4
pip install transformers==4.50.0
pip install peft

pip install moviepy==1.0.0
pip install projectaria-tools
pip install matplotlib
pip install opencv-python
pip install plotly
pip install scipy
pip install h5py
pip install fastdtw
pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install hydra-core

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python setup.py install

pip install jsonlines
pip install httpx
pip install langchain_openai
pip install torchmetrics
pip install termcolor
