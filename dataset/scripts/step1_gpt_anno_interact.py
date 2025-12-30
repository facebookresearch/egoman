# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 1: Interaction Clip Annotation with GPT

Purpose:
    Annotate 5-second egocentric video clips with structured interaction information using GPT.
    This script identifies and labels hand-object interactions with temporal boundaries, stages,
    and semantic descriptions.

Dependencies:
    - Runs first in the pipeline
    - Runs before: Step 2 (valid_interact_filter.py)

Input:
    - 5-second video clips from source datasets (EgoExo4D, Nymeria, HOT3D)
    - Reference atomic action descriptions with timestamps
    - Format: 4 FPS downsampled frames with GPT analysis

Output:
    - Annotated interaction clips with:
        * Intention goal: High-level objective of the interaction sequence
        * Interaction stages with timestamps and descriptions:
            - Approach stage (optional): Hand moving toward object before contact
                * start_time, end_time: Temporal boundaries (seconds)
                * trajectory: {start_point, end_point, shape (linear/curved/arc)}
            - Manipulation stage (required): Direct hand-object interaction
                * start_time, end_time: Temporal boundaries (seconds)
                * verb: Action verb (e.g., "grasp", "open", "pour")
                * object: Target object with brief appearance description
                * hand: Which hand(s) used (left/right/both)
                * trajectory: {start_point, end_point, shape}
        * Atomic description: Natural language description of the interaction
        * Reasoning: Explanation of why this action serves the intention goal
                    and why trajectory follows this pattern

Annotation Rules:
    - Approach stage exists when hand moves to reach manipulation location without object contact
    - If hand already at manipulation location, skip approach and start with manipulation
    - Use precise timestamps from downsampled frames (4 FPS)
    - Object descriptions should be concise, no hand mentions
    - Reasoning must explain: (1) why action serves goal, (2) why trajectory has this pattern

Usage:
    result = process_one_sample(
        video_file="path/to/5s_clip.mp4",
        ref_annos="s1: desc1, s2: desc2, ..." (a dict),
        dataset="egoexo", (or nymeria)
        output_path="path/to/output.pkl",
        temp_dir_root="path/to/temp",
        fps=4
    )

Note: Configure your GPT API credentials before running (see GPT set up section below).
"""

import base64
import glob

import json
import logging
import os
import pickle
import random
import re
import ssl
import time

import cv2

import httpx
import moviepy.editor as mp
from langchain_openai import ChatOpenAI
from PIL import Image


"""GPT set up"""
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"


def get_http_proxy():
    """For Bento: get a httpx client with proxy settings."""
    try:
        from libfb.py.certpathpicker import get_client_credential_paths

        def is_file_readable(file_path: str) -> bool:
            return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

        thrift_cert, thrift_key = get_client_credential_paths()
        if not is_file_readable(thrift_cert) or not is_file_readable(thrift_key):
            raise RuntimeError("Missing key TLS cert settings.")

        ssl_context = ssl.create_default_context()
        ssl_context.load_cert_chain(thrift_cert, thrift_key)

        FWDPROXY_HOSTNAME = "https://fwdproxy"
        FWDPROXY_PORT = 8082
        fwdproxy_url = f"{FWDPROXY_HOSTNAME}:{FWDPROXY_PORT}"
        proxy = httpx.Proxy(fwdproxy_url, ssl_context=ssl_context)

        return httpx.Client(proxy=proxy)
    except Exception as e:
        return None


# Replace with your own gpt api and function
llm = ChatOpenAI(
    model="gpt-4.1",
    base_url="https://api.wearables-ape.io/models/v1/",
    api_key="",
    temperature=0.7,
    http_client=get_http_proxy(),
)
"""End of GPT set up (Modify this part if you want to use your own GPT api and function)"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_frames(video_path, intv_fps=4, temp_dir="../data/anno_temp"):
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / intv_fps)
    images = []

    if not cap.isOpened():
        print("Error: Could not open video.")
        return images

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            image_path = os.path.join(temp_dir, f"frame-{count}.jpg")
            cv2.imwrite(image_path, frame)
            images.append(image_path)

        count += 1

    cap.release()
    return images


def process_one_sample(
    video_file, ref_annos, dataset, output_path, temp_dir_root, fps=4
):
    """
    video_file: path to the 5s video file
    ref_annos: a dict of sorted timestamps within the 5s video and corresponding atomic descriptions formatted as "s1: desc1, s2: desc2, ..."
    dataset: name of source dataset (egoexo, nymeria, or hot3d)
    output_path: path to save the output pickle file
    temp_dir_root: root directory to store temporary files (frames, etc.)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_id = video_file.split("/")[-1].split(".")[0]
    temp_dir = os.path.join(temp_dir_root, video_id)

    # Generate downsampled images
    if not os.path.exists(temp_dir):
        downsample_images = extract_frames(video_file, intv_fps=fps, temp_dir=temp_dir)
    downsample_images = [
        temp_dir + pathi for pathi in os.listdir(temp_dir) if pathi.endswith(".jpg")
    ]
    # Sort downsample_images by the integer frame index i extracted from the filename
    downsample_images.sort(
        key=lambda path: int(path.split("/")[-1].split(".")[0].split("-")[-1])
    )

    # Create messages with the image extracted from the first frame
    content = []
    for i, image_path in enumerate(downsample_images):
        i = int(image_path.split("/")[-1].split(".")[0].split("-")[-1])
        intv = int(30 / fps)
        timestamp = i * 1.0 / fps / intv
        content.append(
            {
                "type": "text",
                "text": f"This is frame {i} at {timestamp} seconds.",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image_path)}",
                },
            }
        )
    messages = [
        {
            "role": "system",
            "content": f"""
                Extract hand-object interactions from video frames.

                **Reference Timestamps**: {ref_annos}

                **Output JSON**:
                {{
                    "intent": "<action goal>",
                    "interactions": [
                        {{
                        "approach": {{ "start_time": <float>, "end_time": <float>, "trajectory": {{ "start_point": "<location>", "end_point": "<location>", "shape": "<linear/curved/arc>" }} }},
                        "manipulation": {{ "start_time": <float>, "end_time": <float>, "verb": "<action>", "object": "<object with short appearance description not mention hand>", "hand": "<left|right|both>", "trajectory": {{ "start_point": "<location>", "end_point": "<location>", "shape": "<linear/curved/arc>" }} }},
                        "atomic_description": "<interaction description>",
                        "reasoning": "<why action serves goal and trajectory pattern>"
                        }}
                    ]
                }}

                **Rules**:
                - approach exists when hand moves to reach manipulation location, no contact until contact the object
                - If hand already contact at manipulation location, skip approach and start with manipulation
                - For each stage, trajectory has three keys: start_point (brief location text), end_point (brief location text), one-word shape (movement pattern)
                - Short reasoning: explain both (1) why action serves intention goal and (2) why trajectory has this pattern
                - Use precise timestamps from frames
                """,
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    # Retry logic for AI response parsing
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ai_message = llm.invoke(messages)
            cur_data["video_file"] = video_file
            cur_data["ai_message"] = ai_message.content
            cur_data["dataset"] = dataset
            cur_data["ref_annos"] = ref_annos
            print(ref_annos)
            print(ai_message.content)

            # Try to parse the response
            eval(ai_message.content)

            # If parsing succeeds, break out of retry loop
            pickle.dump(cur_data, open(output_path, "wb"))
            break

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to get valid response after {max_retries} attempts")
                cur_data["ai_message"] = (
                    f"FAILED_AFTER_{max_retries}_ATTEMPTS: {ai_message.content if 'ai_message' in locals() else 'No response'}"
                )
            else:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(1)  # Brief pause before retry
