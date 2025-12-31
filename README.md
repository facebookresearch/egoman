<div align="center">


## Flowing from Reasoning to Motion: Learning 3D Hand Trajectory Prediction from Egocentric Human Interaction Videos

<p align="center">
  <a href="https://arxiv.org/abs/2512.16907"><img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv"></a>
  <a href="https://egoman-project.github.io/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Website"></a>
</p>

**[Mingfei Chen](https://www.mingfeichen.com/)**<sup>1,2</sup> ·
**[Yifan Wang](https://yifanwang.cc/)**<sup>1</sup> ·
**[Zhengqin Li](https://sites.google.com/view/zhengqinli)**<sup>1</sup> ·
**[Homanga Bharadhwaj](https://homangab.github.io/)**<sup>1</sup> ·
**[Yujin Chen](https://terencecyj.github.io/)**<sup>1</sup> ·
**[Chuan Qin](https://www.linkedin.com/in/chuanqin/)**<sup>1</sup> ·
**[Ziyi Kou](https://ziyikou.me/)**<sup>1</sup> ·
**[Yuan Tian](https://scholar.google.com/citations?user=ddFnVyYAAAAJ&hl=en)**<sup>1</sup> ·
**[Eric Whitmire](https://www.ericwhitmire.com/)**<sup>1</sup> ·
**[Rajinder Sodhi](https://rsodhi.com/)**<sup>1</sup> ·
**[Hrvoje Benko](https://www.hbenko.com/)**<sup>1</sup> ·
**[Eli Shlizerman](https://faculty.washington.edu/shlizee/NW/index.html)**<sup>2</sup> ·
**[Yue Liu](https://openreview.net/profile?id=~Yue_Liu34)**<sup>1</sup>

<sup>1</sup>Meta · <sup>2</sup>University of Washington

</div>

## Abstract

Prior works on 3D hand trajectory prediction are constrained by datasets that decouple motion from semantic supervision and by models that weakly link reasoning and action. To address these, we first present the **EgoMAN dataset**, a large-scale egocentric dataset for interaction stage-aware 3D hand trajectory prediction with **219K 6DoF trajectories** and **3M structured QA pairs** for semantic, spatial, and motion reasoning.

We then introduce the **EgoMAN model**, a reasoning-to-motion framework that links vision–language reasoning and motion generation via a trajectory-token interface. Trained progressively to align reasoning with motion dynamics, our approach yields accurate and stage-aware trajectories with generalization across real-world scenes.

> **Note:** Model weights and processed dataset are not released due to legal and licensing considerations. We provide complete model code and dataset creation scripts in this repo to ensure full reproducibility.

---

## 📑 Overview

| Category | Description |
|----------|-------------|
| **[⚙️ Environment Setup](#️-environment-setup)** | Install dependencies and set up CUDA environment |
| **[🚀 Quick Start & Inference](#-quick-start-single-image-demo)** | Single image and batch trajectory prediction |
| **[📈 Evaluation](#-inference-on-egoman-benchmark)** | Benchmark assessment with trajectory and waypoint metrics |
| **[🔧 Training](#-training-egoman)** | Progressive three-stage training pipeline |
| **[📦 Dataset Creation](#-egoman-dataset-creation)** | Scripts to build EgoMAN dataset from scratch |

---

## ⚙️ Environment Setup

Install dependencies using the provided script (requires CUDA 12.4):

```bash
conda create python=3.12 -n perception_models
conda activate perception_models
bash env_install.sh
```

**Key Dependencies:**
- `torch==2.6.0`, `torchvision==0.21.0`
- `flash-attn==2.7.4`
- `transformers==4.50.0`
- [pytorch3d](https://github.com/facebookresearch/pytorch3d.git)

---

## 🚀 Quick Start: Single Image Demo

Get started quickly by running inference on a single egocentric image. This demo predicts future 3D hand trajectories based on the current scene and a text description of the intended action.

**What you need:**
- An egocentric image (first-person view)
- A text prompt specifying the intended action and the acting hand (e.g., "open refrigerator with left hand")
- (Optional) Past motion data for better performance (e.g., 5 frames of 6DoF hand poses)

**Setup:** Visual features are extracted on-the-fly using DINOv3. First set up DINOv3:
```bash
cd model/semantics_extractor
git clone https://github.com/facebookresearch/dinov3.git
cd ../..
```

**Note:** DINOv3 weights will be automatically downloaded on first run if not found. Alternatively, you can manually download from [here](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and place `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` under `data/weights/`.

Run inference on a single image with text prompt:

```bash
cd model

# Option 1: Run inference WITH past motion (recommended for best performance)
python tools/egoman_demo_vlonly.py \
    --image ../data/examples/open_refrigerator_to_access_contents_with_left_hand.jpg \
    --text "open refrigerator with left hand" \
    --past_motion ../data/examples/open_refrigerator_to_access_contents_with_left_hand+past_motion.npy \
    --model_path ../data/weights/EgoMAN-7B \
    --output_dir ../output \
    --num_samples 3

# Option 2: Run inference WITHOUT past motion (simpler but may degrade performance)
# Note: The --past_motion parameter is optional but recommended for best performance
python tools/egoman_demo_vlonly.py \
    --image ../data/examples/open_refrigerator_to_access_contents_with_left_hand.jpg \
    --text "open refrigerator with left hand" \
    --model_path ../data/weights/EgoMAN-7B \
    --output_dir ../output \
    --num_samples 3
```

**Output:**
- Prediction results: `output/{image_name}_result.pkl`
- Visualizations: `output/visualizations/{image_name}/`

**Note:** Past motion format: `.npy` file with shape `(5, 2, 7)` - 5 frames, 2 hands (left, right), 3D positions + quaternion (x, y, z, qx, qy, qz, qw)



## 📊 Batch Inference on Examples

Process multiple egocentric images in batch mode for efficient trajectory prediction. This workflow demonstrates how to extract visual and text features, run parallel inference, and visualize the predicted hand trajectories.

**Pipeline Overview:**
1. Extract DINOv3 visual features from images
2. (Optional) Extract CLIP text embeddings for action descriptions
3. Run batch inference to generate multiple trajectory samples
4. Visualize results with overlaid trajectories on images

Run batch inference on multiple example images:

```bash
cd model

# Extract DINOv3 visual features from example images
python semantics_extractor/extract_vis_dinov3.py

# (Optional) Extract CLIP features from intention text
python semantics_extractor/extract_text_clip_emb.py

# Infer 3 samples from examples, output saved to output/[MODEL_NAME]-examples.pkl
BATCH_SIZE=1 SAVE_EVERY=50 MODEL_NAME=EgoMAN-7B torchrun --nproc_per_node=1 tools/infer_batch_aria_examples.py

# Visualize the output
python tools/visualize_batch_results.py
```

**Note:** The visualization plots K=3 predicted trajectories by default. You can change K in the function [visualize_predictions](model/tools/visualize_batch_results.py#L46-L52):
```python
visualize_predictions(
    img_dir,
    result_pkl_path=result_pkl_path,
    output_dir=output_dir,
    cam_params_dict=cam_params_dict,
    K=3,  # plot K predicted hand trajectories
)
```


## 📈 Inference on EgoMAN Benchmark

Evaluate the EgoMAN model on our comprehensive benchmark dataset. This benchmark contains challenging egocentric interaction scenarios to assess the model's trajectory prediction accuracy and stage awareness.

**Requirements:**
- Download and place the `egoman_imgs` data folder under `data/`
- Ensure you have multiple GPUs available for parallel processing

**Evaluation Metrics:**
- Trajectory accuracy: ADE, FDE, DTW, ROT
- Waypoint prediction: Contact, Traj-Warp (Traj)

Run the full evaluation pipeline:

```bash
cd model

# Extract DINOv3 visual embeddings, output to data/egomanbench_vis_features.pkl
python semantics_extractor/extract_vis_egomanbench.py

# Run inference with 4 GPUs in parallel
# Output saved to output/[MODEL_NAME]-egomanbench.pkl
BATCH_SIZE=1 SAVE_EVERY=50 MODEL_NAME=EgoMAN-7B torchrun --nproc_per_node=4 tools/infer_batch_egomanbench.py

# Compute trajectory metrics
python tools/egomanbench_metrics.py

# Compute waypoints metrics with shift radius of 0.06 (spatial tolerance of the palm affordance point to wrist position)
python tools/waypoint_metrics.py --shift --shift_radius 0.06
```



## 🔧 Training EgoMAN

Train the EgoMAN model from scratch using our progressive three-stage training approach. This methodology ensures that both reasoning and motion modules are properly aligned for accurate trajectory prediction.

**Training Stages:**
1. **Preprocessing: Feature Extraction** - Extract DINOv3 visual features and CLIP action embeddings from training data
2. **Stage 1: Reasoning Module Pretraining** - Train the vision-language model on semantic, spatial, and motion reasoning tasks
3. **Stage 2: Motion Expert Pretraining** - Train the trajectory decoder on motion dynamics
4. **Stage 3: Joint Training** - Align reasoning with motion generation via the trajectory-token interface

---

### Preprocessing: Feature Extraction (Required Before Training)

Before starting the three-stage training, you must extract visual and text features from your training dataset. This preprocessing step is **critical** as the training dataloaders expect pre-computed features to avoid redundant computation during training.

**Prerequisites:**
- Place your training dataset annotations at:
  - `data/egoman_dataset/egoman_pretrain.pkl` (for reasoning pretraining)
  - `data/egoman_dataset/egoman_finetune.pkl` (for finetuning)
  - `data/egoman_dataset/egoman-test-final.pkl` (for validation, optional)
- Ensure training images are accessible from paths specified in the annotation PKL files

#### Step 1: Extract DINOv3 Visual Features

Extract visual features from all training images using DINOv3. These features are used by all three training stages.

```bash
cd model

# Extract DINOv3 features from training images
python semantics_extractor/extract_vis_dinov3_train.py
```

**Output:**
- `data/egoman_dataset/egoman_dinov3_features.pkl` - Dictionary mapping `{image_path: dinov3_feature_array}`
  - Feature shape: `(1024,)` for DINOv3-L/16

#### Step 2: Extract CLIP Action Embeddings

Extract text embeddings from action phrases using CLIP. These embeddings encode semantic information about intended hand-object interactions.

```bash
cd model

# Extract CLIP embeddings from action phrases
python semantics_extractor/extract_act_emb_clip_train.py
```

**Output:**
- `data/egoman_dataset/act_emb_dict.pkl` - Dictionary mapping `{image_path + "_" + action_phrase: {"text": ..., "emb": ...}}`
  - Embedding shape: `(768,)` for CLIP-L/14
- `data/egoman_dataset/act_emb_val_dict.pkl` (optional) - Same format for validation data
---

### Stage 1: Reasoning Module Pretraining
Train the reasoning module on structured QA pairs to develop understanding of egocentric interactions, object relationships, and motion semantics.

```bash
cd model
sh train_scripts/reason_pretrain.sh
```
---

### Stage 2: Motion Expert Pretraining
Train the motion generation module to learn realistic hand trajectory dynamics and interaction patterns.

```bash
cd model
sh train_scripts/motion_pretrain.sh
```
---

### Stage 3: Joint Training with Trajectory-Token Interface
Fine-tune both modules together with the trajectory-token interface to enable seamless reasoning-to-motion transfer.

```bash
cd model
sh train_scripts/joint_finetune.sh
```
---

## 📦 EgoMAN Dataset Creation

Build the EgoMAN dataset from scratch using our automated pipeline. This process transforms raw egocentric videos into a comprehensive dataset with 219K trajectories and 3M QA pairs.

**Overview:**
The dataset creation pipeline consists of 5 automated steps that progressively process raw video data into structured interaction episodes with semantic annotations and 6DoF hand trajectories.

**Source Datasets Required:**
- [EgoExo4D](https://docs.ego-exo4d-data.org/download/) - Large-scale egocentric video dataset
- [Nymeria Dataset](https://github.com/facebookresearch/nymeria_dataset) - Egocentric interaction videos
- [HOT3D](https://github.com/facebookresearch/hot3d) - 3D hand-object tracking dataset

**Setup:**
Place downloaded source datasets under `data/egoman_dataset/`. You'll need to configure your own GPT API credentials and update data paths in the scripts.

We provide scripts with prompts to create the EgoMAN dataset. Please replace the GPT call function and api with your own.

Please download the source datasets following their own repo: [EgoExo4D](https://docs.ego-exo4d-data.org/download/), [Nymeria Dataset](https://github.com/facebookresearch/nymeria_dataset), [HOT3D](https://github.com/facebookresearch/hot3d). The data should be put under data/egoman_dataset. We recommend you update the source data path, output data path and temp file paths in these scripts based on your own case.

### Step 1: Interaction Clip Annotation
Annotate 5-second video clips with interaction stages (approach, manipulation) using GPT-4.1. This step identifies key interaction moments and labels them with text descriptions, timestamps, and reasoning.

**Output:** Annotated interaction clips with stage labels and descriptions

Annoate 5s video clips with interaction stages (approach, manipulation) including text, timestamps, reason, etc.

Annotation function with GPT4.1 in script: [dataset/scripts/step1_gpt_anno_interact.py](dataset/scripts/step1_gpt_anno_interact.py).

### Step 2: Valid Interaction Filtering
Apply rule-based and GPT-powered filters to remove invalid annotations. This step ensures interactions are realistic, properly timed, and semantically meaningful. Also generates high-level intention summaries.

**Filters Applied:**
- Duration constraints (not too short/long)
- Semantic relevance checks
- Realism validation
- Annotation quality assessment

We use rules and GPT to filter out invalid interactions that are wrongly annotated, too short/long, not relevant, unrealistic. We also summarize the high level intention goal using GPT.

Filtering functions in script: [dataset/scripts/step2_valid_interact_filter.py](dataset/scripts/step2_valid_interact_filter.py).

### Step 3: Non-Numeric QA Generation
Generate diverse non-numeric question-answer pairs for semantic, spatial, and motion reasoning. Each valid interaction produces multiple QA pairs covering object recognition, spatial relationships, motion patterns, and interaction stages.

**Output QA Categories:**
- Current intention goals
- Which hand will be used
- What action will occur
- What object will be manipulated
- Hand trajectory descriptions
- Interaction stage information
- Reasoning about why actions occur

Generate non-numeric QA pairs using GPT for each valid interaction item after filtering in Step 2.

Generator function in script: [dataset/scripts/step3_gpt_qa_generator.py](dataset/scripts/step3_gpt_qa_generator.py).

### Step 4: Hand Trajectory Extraction (6DoF)
Extract 6DoF hand trajectories (3D position + quaternion orientation) from the source datasets. This step processes hand tracking data and aligns it with the annotated interaction clips from Step 2.

**Trajectory Format:**
- Position: (x, y, z) in meters, camera-relative coordinates
- Orientation: Quaternion (qx, qy, qz, qw)
- Frequency: 10 FPS
- Smooth interpolation for missing frames

**Output:**
- 6DoF hand trajectories aligned with interaction clips
- Projected 2D wrist positions in image coordinates
- Head pose trajectory for context
- Camera transformation matrices

Extract 6DoF hand trajectory (3D location and quaternion) from MPS hand tracking data.

**Note:** For EgoExo4D, we re-run the MPS Hand Tracking Service and place the hand tracking result for each take under: `data/egoman_dataset/egoexo/vrs_list/[take_name]/hand_tracking/`

Script for EgoExo and Nymeria: [dataset/scripts/step4_6dof_traj_process.py](dataset/scripts/step4_6dof_traj_process.py).

Script for HOT3D: [dataset/scripts/step4_6dof_traj_process_hot3d.py](dataset/scripts/step4_6dof_traj_process_hot3d.py).

### Step 5: Numeric QA Generation (Reasoning + Motion)
**Important:** This step runs AFTER Step 4 and requires trajectory data with 6DoF hand poses.

Generate numeric question-answer pairs that require quantitative reasoning about hand trajectories. This script creates QA pairs with numeric answers (3D positions, timestamps, quaternions) for both pretraining and finetuning the reasoning module.

**Output QA Types:**
- **Pretraining QA** (diverse individual questions):
  - Temporal: "When will the hand approach/complete manipulation?"
  - Spatial: "What will be the 3D position of the [hand] at [stage]?"
  - Spatiotemporal: "When and where will the hand make contact?"
  - Action semantic identification: "What is the next hand-object interaction?"

- **Finetuning QA** (full trajectory prediction):
  - Question: "Where will the hands move to [intention]?<HOI_QUERY>"
  - Answer: "<ACT><START><CONTACT><END>" with full trajectory data
  - Includes past motion context (5 frames of historical poses)

**Numeric Answer Format:**
- Special tokens: `<ACT>`, `<START>`, `<CONTACT>`, `<END>`, etc.
- 11-dimensional vectors encoding timestamps, 3D positions, and 2D projections

Generator function in script: [dataset/scripts/step5_reason_numeric_qa_generator.py](dataset/scripts/step5_reason_numeric_qa_generator.py).

Combine the outputs from Step 3 (non-numeric QA) and Step 5 (numeric QA) to form the complete reasoning dataset for training.


### Step 6: High Quality Trajectory Filtering
Final quality control to ensure only high-quality, physically plausible trajectories are included in the finetuning dataset and evaluation benchmark.

**Quality Criteria:**
- Smooth motion without abrupt jumps
- Physically plausible hand movements
- Consistent with visual observations
- Proper alignment with interaction stages

Filter out low quality trajectories:
- By rules: [dataset/scripts/step6_traj_quality_filter_rules.py](dataset/scripts/step6_traj_quality_filter_rules.py)
- By GPT: [dataset/scripts/step6_traj_quality_filter_gpt.py](dataset/scripts/step6_traj_quality_filter_gpt.py)




## 📄 License

The majority of EgoMAN is licensed under [CC-BY-NC](LICENSE), however portions of the project adapted function code are available under separate license terms: [QwenVL3](https://github.com/QwenLM/Qwen3-VL) and [FastChat](https://github.com/lm-sys/FastChat) are licensed under the Apache 2.0 license.



## 📚 Citation

If you find EgoMAN useful in your research, please consider citing:

```bibtex
@misc{chen2025flowingreasoningmotionlearning,
      title={Flowing from Reasoning to Motion: Learning 3D Hand Trajectory Prediction from Egocentric Human Interaction Videos},
      author={Mingfei Chen and Yifan Wang and Zhengqin Li and Homanga Bharadhwaj and Yujin Chen and Chuan Qin and Ziyi Kou and Yuan Tian and Eric Whitmire and Rajinder Sodhi and Hrvoje Benko and Eli Shlizerman and Yue Liu},
      year={2025},
      eprint={2512.16907},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16907},
}
```
