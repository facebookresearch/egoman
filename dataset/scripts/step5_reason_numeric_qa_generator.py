# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 5: Numeric QA Generation

Purpose:
    Generate numeric question-answer pairs that require quantitative reasoning about hand trajectories.
    This script creates QA pairs with numeric answers (3D positions, timestamps, quaternions) for both
    pretraining and finetuning the reasoning module.

Dependencies:
    - Runs after: Step 4 (6dof_traj_process.py) - REQUIRES trajectory data with 6DoF hand poses

Input:
    - Processed annotations from Step 4 containing:
        * 6DoF hand trajectories (position + quaternion)
        * Interaction stage information (approach, manipulation)
        * Temporal waypoints (start, contact, end)
        * Projected 2D hand positions in image space
        * Intention goals and action descriptions

Output:
    - Numeric QA pairs for pretraining (diverse individual questions):
        * Temporal questions: "When will the hand approach/complete manipulation?"
        * Spatial questions: "What will be the 3D position of the [hand] at [stage]?"
        * Spatiotemporal questions: "When and where will the hand make contact?"
        * Action identification: "What is the next hand-object interaction?"

    - Full-template QA for finetuning (complete trajectory prediction):
        * Question: "Where will the hands move to [intention]?<HOI_QUERY>"
        * Answer: "<ACT><START><CONTACT><END>" with full trajectory data
        * Includes past motion context (5 frames of historical poses)

Numeric Answer Format:
    - Each answer uses special tokens (<ACT>, <START>, <CONTACT>, <END>, <HAND_LEFT>, etc.)
    - Numeric values encoded as 11-dimensional vectors:
        * [0]: timestamp (seconds)
        * [1:4]: left hand 3D position (x, y, z in meters)
        * [4:7]: right hand 3D position (x, y, z in meters)
        * [7:9]: left hand 2D position (pixel coordinates)
        * [9:11]: right hand 2D position (pixel coordinates)
        * -1000: placeholder for unused dimensions

Training Strategy:
    - Pretraining: Diverse QA pairs with varied question formats and answer types
    - Finetuning: Full-context trajectory prediction with past motion
    - 30% of pretraining questions include past motion context for motion reasoning

Usage:
    result = generate_QA_dataset_per_data(cur_data)
    pretrain_numeric_qa_list, finetune_qa_item = result

    Where cur_data is a single annotation from Step 4 with trajectory data.

Dataset Compilation:
    Combine Step 5 (numeric QA) with Step 3 (non-numeric QA) to build the complete reasoning dataset for both pretraining and finetuning.
    For finetuning, we use only the full-template QA format with past motion context, applied after Step 6 filters for high-quality trajectories.
"""

import copy
import functools
import json
import os
import pickle
import random
import re

from typing import Dict, final, List, Union

import numpy as np
import pytorch3d.transforms as pt
import torch

from scipy.spatial.transform import Rotation as R

from tqdm import tqdm


def generate_pretrain_numeric_QA(
    last_frame_traj,
    cur_data,
):
    intention_list = [
        f"Given the intention to {cur_data['intention']},",
        f"To achieve {cur_data['intention']},",
        f"In order to {cur_data['intention']},",
        f"For the purpose of {cur_data['intention']},",
        f"As part of {cur_data['intention']},",
        f"When attempting to {cur_data['intention']},",
    ]
    qa_list = []

    # Meta info
    intent = cur_data["intention"]
    verb = last_frame_traj["verb"]
    obj = last_frame_traj["object"]
    hand = last_frame_traj["hand"]

    start = last_frame_traj["start"]
    contact = last_frame_traj.get("contact", None)
    end = last_frame_traj["end"]

    act_phrase = f"{hand} hand {verb} {obj}"
    cur_question = random.choice(
        [
            f"{random.choice(intention_list)} what will be the next hand-object interaction?",
            f"{random.choice(intention_list)} describe the next action that will happen.",
            f"{random.choice(intention_list)} what is the manipulation event expected to occur?",
            f"{random.choice(intention_list)} what action will the hand perform on the object?",
            f"{random.choice(intention_list)} summarize the upcoming interaction.",
            f"{random.choice(intention_list)} what high-level manipulation will occur next?",
        ]
    )
    qa_list.append(
        {
            "q": cur_question,
            "a": "<ACT>",
            "field": ["act.phrase"],
            "value": [act_phrase],
        }
    )

    # Hand QA
    if hand == "both":
        answer = "<HAND_BOTH>"
    elif hand == "left":
        answer = "<HAND_LEFT>"
    else:
        answer = "<HAND_RIGHT>"
    cur_question = random.choice(
        [
            f"{random.choice(intention_list)} which hand will be used to {verb} the {obj}?",
            f"{random.choice(intention_list)} what hand will be used?",
        ]
    )
    qa_list.append(
        {
            "q": cur_question,
            "a": answer,
            "field": ["hand"],
            "value": [hand],
        }
    )

    # Contact time QA
    cur_question = random.choice(
        [
            f"{random.choice(intention_list)} when will the hand approach the {obj}?",
            f"{random.choice(intention_list)} when will the hand approach the target object?",
        ]
    )

    if contact and "time" in contact:
        value_vec = [-1000] * 11
        value_vec[0] = contact["time"]
        qa_list.append(
            {
                "q": cur_question,
                "a": "<CONTACT>",
                "field": ["contact.time"],
                "value": [value_vec],
            }
        )

    # End time QA
    cur_question = random.choice(
        [
            f"{random.choice(intention_list)} when will the hand manipulation end?",
            f"{random.choice(intention_list)} when will the hand finish {verb} the {obj}?",
        ]
    )

    if "time" in end:
        value_vec = [-1000] * 11
        value_vec[0] = end["time"]
        qa_list.append(
            {
                "q": cur_question,
                "a": "<END>",
                "field": ["end.time"],
                "value": [value_vec],
            }
        )
    cur_question = random.choice(
        [
            f"{random.choice(intention_list)} when will the hand manipulation start and end?",
            f"{random.choice(intention_list)} when will the hand start and end {verb} the {obj}?",
        ]
    )

    if contact and "time" in contact and "time" in end:
        value_vec = [-1000] * 11
        value_vec1 = [-1000] * 11
        value_vec[0] = contact["time"]
        value_vec1[0] = end["time"]
        qa_list.append(
            {
                "q": cur_question,
                "a": "<CONTACT><END>",
                "field": ["contact.time", "end.time"],
                "value": [value_vec, value_vec1],
            }
        )

    # Contact 3D position QA (per hand)
    if hand == "both":
        hand_candidates = ["left", "right"]
    elif hand == "left":
        hand_candidates = ["left"]
    else:
        hand_candidates = ["right"]
    if contact:
        for h in hand_candidates:
            if h == "left":
                field = ["contact.l"]
                contact_key = "l"
                value_vec = [-1000] * 11
                value_vec[1:4] = contact[contact_key].copy()
                if contact[f"{contact_key}2d"] is not None:
                    value_vec[7:9] = contact[f"{contact_key}2d"].copy()
            else:
                field = ["contact.r"]
                contact_key = "r"
                value_vec = [-1000] * 11
                value_vec[4:7] = contact[contact_key].copy()
                if contact[f"{contact_key}2d"] is not None:
                    value_vec[9:] = contact[f"{contact_key}2d"].copy()
            cur_question = random.choice(
                [
                    f"{random.choice(intention_list)} what will be the 3D position of the {h} hand at contact?",
                    f"{random.choice(intention_list)} where will be the {h} hand located in 3D space at the end time of approach stage?",
                    f"{random.choice(intention_list)} what will be the 3D spatial coordinates of the {h} hand after approaching the target object {obj}?",
                    f"{random.choice(intention_list)} what will be the {h} hand's 3D position at contact time {contact['time']}s?",
                ]
            )

            qa_list.append(
                {
                    "q": cur_question,
                    "a": f"<CONTACT>",
                    "field": field,
                    "value": [value_vec],
                }
            )
        if hand == "both":
            cur_question = random.choice(
                [
                    f"{random.choice(intention_list)} what will be the 3D position of the both hands at contact?",
                    f"{random.choice(intention_list)} where will be both hands located in 3D space at the end time of approach stage?",
                    f"{random.choice(intention_list)} what will be the 3D spatial coordinates of both hands after approaching the target object {obj}?",
                    f"{random.choice(intention_list)} what will be both hands' 3D position at contact time {contact['time']}s?",
                ]
            )
            value_vec = [-1000] * 11
            value_vec[1:4] = contact["l"].copy()
            value_vec[4:7] = contact["r"].copy()
            if contact["l2d"] is not None:
                value_vec[7:9] = contact["l2d"].copy()
            if contact["r2d"] is not None:
                value_vec[9:] = contact["r2d"].copy()
            qa_list.append(
                {
                    "q": cur_question,
                    "a": f"<CONTACT>",
                    "field": ["contact.lr"],
                    "value": [value_vec],
                }
            )

    # End 3D position QA (per hand)
    for h in hand_candidates:
        if h == "left":
            field = ["end.l"]
            contact_key = "l"
            value_vec = [-1000] * 11
            value_vec[1:4] = end[contact_key].copy()
            if end[f"{contact_key}2d"] is not None:
                value_vec[7:9] = end[f"{contact_key}2d"].copy()
        else:
            field = ["end.r"]
            contact_key = "r"
            value_vec = [-1000] * 11
            value_vec[4:7] = end[contact_key].copy()
            if end[f"{contact_key}2d"] is not None:
                value_vec[9:] = end[f"{contact_key}2d"].copy()

        cur_question = random.choice(
            [
                f"{random.choice(intention_list)} what will be the 3D position of the {h} hand at manipulation end?",
                f"{random.choice(intention_list)} where will be the {h} hand located in 3D space at the end time of manipulation stage?",
                f"{random.choice(intention_list)} what will be the 3D spatial coordinates of the {h} hand after completing {verb} the {obj}?",
                f"{random.choice(intention_list)} what will be the {h} hand's 3D position at manipulation end time {end['time']}s?",
            ]
        )

        qa_list.append(
            {
                "q": cur_question,
                "a": "<END>",
                "field": field,
                "value": [value_vec],
            }
        )

        if hand == "both":
            cur_question = random.choice(
                [
                    f"{random.choice(intention_list)} what will be the 3D position of both hands at manipulation end?",
                    f"{random.choice(intention_list)} where will be both hands located in 3D space at the end time of manipulation stage?",
                    f"{random.choice(intention_list)} what will be the 3D spatial coordinates of both hands after completing {verb} the {obj}?",
                    f"{random.choice(intention_list)} what will be both hands' 3D position at manipulation end time {end['time']}s?",
                ]
            )
            value_vec = [-1000] * 11
            value_vec[1:4] = end["l"].copy()
            value_vec[4:7] = end["r"].copy()
            if end["l2d"] is not None:
                value_vec[7:9] = end["l2d"].copy()
            if end["r2d"] is not None:
                value_vec[9:] = end["r2d"].copy()
            qa_list.append(
                {
                    "q": cur_question,
                    "a": f"<END>",
                    "field": ["end.lr"],
                    "value": [value_vec],
                }
            )

    # Start 3D position QA (per hand)
    for h in ["l", "r"]:
        if h == "l":
            cur_hand = "left"
            value_vec = [-1000] * 11
            value_vec[1:4] = start["l"].copy()
            if start["l2d"] is not None:
                value_vec[7:9] = start["l2d"].copy()
        else:
            cur_hand = "right"
            value_vec = [-1000] * 11
            value_vec[4:7] = start["r"].copy()
            if start["r2d"] is not None:
                value_vec[9:] = start["r2d"].copy()
        cur_question = random.choice(
            [
                f"{random.choice(intention_list)} where is the current 3D position of the {cur_hand} hand?",
                f"{random.choice(intention_list)} what are the 3D spatial coordinates of the {cur_hand} hand at the beginning?",
            ]
        )
        qa_list.append(
            {
                "q": cur_question,
                "a": "<START>",
                "field": [f"start.{h}"],
                "value": [value_vec],
            }
        )
    cur_question = random.choice(
        [
            f"{random.choice(intention_list)} where are the current 3D position of both hands?",
            f"{random.choice(intention_list)} what are the 3D spatial coordinates of both hands at the beginning?",
        ]
    )
    value_vec = [-1000] * 11
    value_vec[1:4] = start["l"].copy()
    if start["l2d"] is not None:
        value_vec[7:9] = start["l2d"].copy()
    value_vec[4:7] = start["r"].copy()
    if start["r2d"] is not None:
        value_vec[9:] = start["r2d"].copy()
    qa_list.append(
        {
            "q": cur_question,
            "a": "<START>",
            "field": ["start.lr"],
            "value": [value_vec],
        }
    )

    # Joint QA: contact & end 3D positions
    if contact:
        for h in hand_candidates:
            if h == "left":
                value_vec = [-1000] * 11
                value_vec[1:4] = contact["l"].copy()
                if contact["l2d"] is not None:
                    value_vec[7:9] = contact["l2d"].copy()
                value_vec1 = [-1000] * 11
                value_vec1[1:4] = end["l"].copy()
                if end["l2d"] is not None:
                    value_vec1[7:9] = end["l2d"].copy()
                field = ["contact.l", "end.l"]
                value = [value_vec, value_vec1]
            else:
                value_vec = [-1000] * 11
                value_vec[4:7] = contact["r"].copy()
                if contact["r2d"] is not None:
                    value_vec[9:] = contact["r2d"].copy()
                value_vec1 = [-1000] * 11
                value_vec1[4:7] = end["r"].copy()
                if end["r2d"] is not None:
                    value_vec1[9:] = end["r2d"].copy()

                field = ["contact.r", "end.r"]
                value = [value_vec, value_vec1]
            cur_question = random.choice(
                [
                    f"{random.choice(intention_list)} provide the 3D positions of the {h} hand at the start and end of the manipulation.",
                    f"{random.choice(intention_list)} provide the 3D positions of the {h} hand at the start and end of {verb} the {obj}.",
                    f"{random.choice(intention_list)} provide the 3D positions of the {h} hand at the end of approach and the end of manipulation.",
                    f"{random.choice(intention_list)} provide the 3D positions of the {h} hand at the end of approach and the end of {verb} the {obj}.",
                    f"{random.choice(intention_list)} What will be the 3D spatial coordinates of the {h} hand at the end of approach and the end of manipulation?",
                    f"{random.choice(intention_list)} What will be the 3D spatial coordinates of the {h} hand at the end of approach and the end of {verb} the {obj}?",
                    f"{random.choice(intention_list)} What will be the 3D spatial coordinates of the {h} hand at the start and the end of manipulation?",
                    f"{random.choice(intention_list)} What will be the 3D spatial coordinates of the {h} hand at the start and the end of {verb} the {obj}?",
                    f"{random.choice(intention_list)} What will be the 3D spatial coordinates of the {h} hand at the start (at {contact['time']}s) and the end (at {end['time']}s) of manipulation?",
                    f"{random.choice(intention_list)} What will be the 3D spatial coordinates of the {h} hand at the start (at {contact['time']}s) and the end (at {end['time']}s) of {verb} the {obj}?",
                ]
            )
            qa_list.append(
                {
                    "q": cur_question,
                    "a": "<CONTACT><END>",
                    "field": field,
                    "value": value,
                }
            )

        if contact:
            for h in hand_candidates:
                field = [f"contact.time{h[0]}"]  # time + 3D for that hand
                if h == "left":
                    value_vec = [-1000] * 11
                    value_vec[0] = contact["time"]
                    value_vec[1:4] = contact["l"].copy()
                    if contact["l2d"] is not None:
                        value_vec[7:9] = contact["l2d"].copy()
                elif h == "right":
                    value_vec = [-1000] * 11
                    value_vec[0] = contact["time"]
                    value_vec[4:7] = contact["r"].copy()
                    if contact["r2d"] is not None:
                        value_vec[9:] = contact["r2d"].copy()
                qa_list.append(
                    {
                        "q": random.choice(
                            [
                                f"{random.choice(intention_list)} when and where will the {h} hand make contact with the {obj}?",
                                f"{random.choice(intention_list)} provide the contact time and 3D position of the {h} hand.",
                                f"{random.choice(intention_list)} at what time and location will the {h} hand reach the object?",
                                f"{random.choice(intention_list)} at what time and location will the {h} hand reach the {obj}?",
                                f"{random.choice(intention_list)} estimate both the moment and 3D coordinates of the {h} hand at contact.",
                            ]
                        ),
                        "a": "<CONTACT>",  #  single special token
                        "field": field,
                        "value": [value_vec],  #  1 + 3 dims
                    }
                )
            if hand == "both":
                field = [
                    "contact.timelr",
                ]  # time + 3D for both hands
                value_vec = [-1000] * 11
                value_vec[0] = contact["time"]
                value_vec[1:4] = contact["l"].copy()
                if contact["l2d"] is not None:
                    value_vec[7:9] = contact["l2d"].copy()
                value_vec[4:7] = contact["r"].copy()
                if contact["r2d"] is not None:
                    value_vec[9:] = contact["r2d"].copy()
                qa_list.append(
                    {
                        "q": random.choice(
                            [
                                f"{random.choice(intention_list)} when and where will both hands make contact with the {obj}?",
                                f"{random.choice(intention_list)} provide the contact time and 3D positions of both hands.",
                                f"{random.choice(intention_list)} at what time and location will both hands reach the object?",
                                f"{random.choice(intention_list)} at what time and location will both hands reach the {obj}?",
                                f"{random.choice(intention_list)} estimate both the moment and 3D coordinates of  both hands at contact.",
                            ]
                        ),
                        "a": "<CONTACT>",  # single special token
                        "field": field,
                        "value": [value_vec],  #
                    }
                )

        # ---------------------------------------------
        # Time + Position joint QA (End)
        for h in hand_candidates:
            field = [f"end.time{h[0]}"]  # time + 3D for that hand
            if h == "left":
                value_vec = [-1000] * 11
                value_vec[0] = end["time"]
                value_vec[1:4] = end["l"].copy()
                if end["l2d"] is not None:
                    value_vec[7:9] = end["l2d"].copy()
            elif h == "right":
                value_vec = [-1000] * 11
                value_vec[0] = end["time"]
                value_vec[4:7] = end["r"].copy()
                if end["r2d"] is not None:
                    value_vec[9:] = end["r2d"].copy()
            qa_list.append(
                {
                    "q": random.choice(
                        [
                            f"{random.choice(intention_list)} when and where will the {h} hand complete the {verb}?",
                            f"{random.choice(intention_list)} provide the manipulation end time and final 3D location of the {h} hand.",
                            f"{random.choice(intention_list)} at what moment and position will the {h} hand finish {verb} the {obj}?",
                            f"{random.choice(intention_list)} estimate the final time and spatial coordinates for the {h} hand when manipulation ends.",
                        ]
                    ),
                    "a": "<END>",  #  single special token
                    "field": field,
                    "value": [value_vec],  #  1 + 3 dims
                }
            )
        if hand == "both":
            field = [
                "end.timelr",
            ]  # time + 3D for both hands
            value_vec = [-1000] * 11
            value_vec[0] = end["time"]
            value_vec[1:4] = end["l"].copy()
            if end["l2d"] is not None:
                value_vec[7:9] = end["l2d"].copy()
            value_vec[4:7] = end["r"].copy()
            if end["r2d"] is not None:
                value_vec[9:] = end["r2d"].copy()
            qa_list.append(
                {
                    "q": random.choice(
                        [
                            f"{random.choice(intention_list)} when and where will both hands complete the {verb}?",
                            f"{random.choice(intention_list)} provide the manipulation end time and final 3D locations of both hands.",
                            f"{random.choice(intention_list)} at what moment and position will both hands finish {verb} the {obj}?",
                            f"{random.choice(intention_list)} estimate the final time and spatial coordinates for both hands when manipulation ends.",
                        ]
                    ),
                    "a": "<END>",  #  single special token
                    "field": field,
                    "value": [value_vec],  #
                }
            )

        # ---------------------------------------------
        # Contact + End joint QA (time + loc together)
        if contact:
            for h in hand_candidates:
                field = [f"contact.time{h[0]}", f"end.time{h[0]}"]
                if h == "left":
                    value_vec = [-1000] * 11
                    value_vec[0] = contact["time"]
                    value_vec[1:4] = contact["l"].copy()
                    if contact["l2d"] is not None:
                        value_vec[7:9] = contact["l2d"].copy()
                    value_vec1 = [-1000] * 11
                    value_vec1[0] = end["time"]
                    value_vec1[1:4] = end["l"].copy()
                    if end["l2d"] is not None:
                        value_vec1[7:9] = end["l2d"].copy()
                elif h == "right":
                    value_vec = [-1000] * 11
                    value_vec[0] = contact["time"]
                    value_vec[4:7] = contact["r"].copy()
                    if contact["r2d"] is not None:
                        value_vec[9:] = contact["r2d"].copy()
                    value_vec1 = [-1000] * 11
                    value_vec1[0] = end["time"]
                    value_vec1[4:7] = end["r"].copy()
                    if end["r2d"] is not None:
                        value_vec1[9:] = end["r2d"].copy()
                qa_list.append(
                    {
                        "q": random.choice(
                            [
                                f"{random.choice(intention_list)} provide both the contact and manipulation end information: times and 3D positions of the {h} hand.",
                                f"{random.choice(intention_list)} give the full spatiotemporal profile of the {h} hand at start and at the end of manipulation.",
                                f"{random.choice(intention_list)} when and where will the {h} hand make contact and finish {verb} the {obj}?",
                                f"{random.choice(intention_list)} return the time and 3D coordinates of the {h} hand for both contact and manipulation completion.",
                            ]
                        ),
                        "a": "<CONTACT><END>",  #  one per event, each covers 4 dims (time + 3D)
                        "field": field,
                        "value": [value_vec, value_vec1],
                    }
                )
            if hand == "both":
                field = [
                    "contact.timelr",
                    "end.timelr",
                ]
                value_vec = [-1000] * 11
                value_vec[0] = contact["time"]
                value_vec[1:4] = contact["l"].copy()
                if contact["l2d"] is not None:
                    value_vec[7:9] = contact["l2d"].copy()
                value_vec[4:7] = contact["r"].copy()
                if contact["r2d"] is not None:
                    value_vec[9:] = contact["r2d"].copy()
                value_vec1 = [-1000] * 11
                value_vec1[0] = end["time"]
                value_vec1[1:4] = end["l"].copy()
                if end["l2d"] is not None:
                    value_vec1[7:9] = end["l2d"].copy()
                value_vec1[4:7] = end["r"].copy()
                if end["r2d"] is not None:
                    value_vec1[9:] = end["r2d"].copy()

                qa_list.append(
                    {
                        "q": random.choice(
                            [
                                f"{random.choice(intention_list)} provide both the contact and manipulation end information: times and 3D positions of both hands.",
                                f"{random.choice(intention_list)} give the full spatiotemporal profile of both hands at start and at the end of manipulation.",
                                f"{random.choice(intention_list)} when and where will both hands make contact and finish {verb} the {obj}?",
                                f"{random.choice(intention_list)} return the time and 3D coordinates of both hands for both contact and manipulation completion.",
                            ]
                        ),
                        "a": "<CONTACT><END>",  #  one per event, each covers 4 dims (time
                        "field": field,
                        "value": [value_vec, value_vec1],
                    }
                )

    return qa_list


def generate_QA_dataset_per_data(cur_data):
    """
    Generate numeric QA training dataset for a single data.
    Input:
        cur_data (dict): a single data item from traj process of EgoExo4D and Nymeria Dataset (step4_6dof_traj_process.py)
    Output:
        pretrain_numeric_qa_list (list): a list of numeric QA pairs for pretraining.
        finetune_qa_item (dict): QA item for finetuning from the current data.
    Combining the numeric QA dataset and the non-numeric QA dataset from step3 (step3_generate_QA_dataset.py), we can get the final QA dataset for the current data.
    """
    video_id = cur_data["video"].split("/")[-2]
    if "pose_data" not in cur_data:
        return [], None
    cur_anno = cur_data["pose_data"]
    if cur_data["dataset"] == "egoexo":
        left_quat = np.array(cur_anno["left_hand_data"]["wrist_quat_transformed"])
        right_quat = np.array(cur_anno["right_hand_data"]["wrist_quat_transformed"])
        left_pos = np.array(cur_anno["left_hand_data"]["wrist_pos_transformed"])
        right_pos = np.array(cur_anno["right_hand_data"]["wrist_pos_transformed"])
    else:
        left_quat = np.array(cur_anno["wrist_quat"])[..., :4]
        right_quat = np.array(cur_anno["wrist_quat"])[..., 4:]
        left_pos = np.array(cur_anno["wrist_pose"])[..., :3]
        right_pos = np.array(cur_anno["wrist_pose"])[..., 3:]

    left_wrist_pos2d = cur_anno["project_left_wrist"]
    right_wrist_pos2d = cur_anno["project_right_wrist"]

    if cur_data["interact"]["two_stage"] is False:
        contact_waytime = None
    else:
        cur_data["interact"]["approach"]["end_time"] = round(
            cur_data["interact"]["approach"]["end_time"] / 4.5, 3
        )
        contact_waytime = int(float(cur_data["interact"]["approach"]["end_time"]) * 10)
    cur_data["interact"]["manipulation"]["end_time"] = round(
        cur_data["interact"]["manipulation"]["end_time"] / 4.5, 3
    )
    cur_data["interact"]["manipulation"]["start_time"] = round(
        cur_data["interact"]["manipulation"]["start_time"] / 4.5, 3
    )
    end_waytime = int(float(cur_data["interact"]["manipulation"]["end_time"]) * 10)

    # start
    start_waypoint = {
        "time": 0.0,
        "left": {
            "2d": (
                list(map(int, np.round(left_wrist_pos2d[5])))
                if (
                    left_wrist_pos2d[5] is not None
                    and np.all(left_wrist_pos2d[5] is not None)
                )
                else None
            ),
            "3d": np.round(left_pos[5], 3).tolist(),
        },
        "right": {
            "2d": (
                list(map(int, np.round(right_wrist_pos2d[5])))
                if (
                    right_wrist_pos2d[5] is not None
                    and np.all(right_wrist_pos2d[5] is not None)
                )
                else None
            ),
            "3d": np.round(right_pos[5], 3).tolist(),
        },
    }
    visible_start = [0, 0]
    if start_waypoint["left"]["2d"] is not None:
        visible_start[0] = 1
    if start_waypoint["right"]["2d"] is not None:
        visible_start[1] = 1

    # contact
    if contact_waytime is None:
        contact_waypoint = None
    else:
        contact_idx = min(contact_waytime + 5, len(left_wrist_pos2d) - 1)
        contact_waypoint = {
            "time": round((contact_idx - 5) / 10.0 / 4.5, 3),
            "left": {
                "2d": (
                    list(map(int, np.round(left_wrist_pos2d[contact_idx])))
                    if (
                        left_wrist_pos2d[contact_idx] is not None
                        and np.all(left_wrist_pos2d[contact_idx] is not None)
                    )
                    else None
                ),
                "3d": np.round(left_pos[contact_idx], 3).tolist(),
            },
            "right": {
                "2d": (
                    list(map(int, np.round(right_wrist_pos2d[contact_idx])))
                    if (
                        right_wrist_pos2d[contact_idx] is not None
                        and np.all(right_wrist_pos2d[contact_idx] is not None)
                    )
                    else None
                ),
                "3d": np.round(right_pos[contact_idx], 3).tolist(),
            },
        }
        visible_contact = [0, 0]
        if contact_waypoint["left"]["2d"] is not None:
            visible_contact[0] = 1
        if contact_waypoint["right"]["2d"] is not None:
            visible_contact[1] = 1

    # end
    end_waypoint = {
        "time": round((len(left_wrist_pos2d) - 6) / 10 / 4.5, 3),
        "left": {
            "2d": (
                list(map(int, np.round(left_wrist_pos2d[-1])))
                if (
                    left_wrist_pos2d[-1] is not None
                    and np.all(left_wrist_pos2d[-1] is not None)
                )
                else None
            ),
            "3d": np.round(left_pos[-1], 3).tolist(),
        },
        "right": {
            "2d": (
                list(map(int, np.round(right_wrist_pos2d[-1])))
                if (
                    right_wrist_pos2d[-1] is not None
                    and np.all(right_wrist_pos2d[-1] is not None)
                )
                else None
            ),
            "3d": np.round(right_pos[-1], 3).tolist(),
        },
    }

    past_motion = "<past_motion><past_motion><past_motion><past_motion><past_motion>"
    past_motion_vec = np.concatenate(
        [np.round(left_pos[:5], 3), np.round(right_pos[:5], 3)], axis=1
    )

    contact_time = (
        contact_waypoint["time"]
        if contact_waypoint is not None
        else start_waypoint["time"]
    )
    end_time = end_waypoint["time"]
    if contact_waypoint is not None:
        last_frame_traj = {
            "hand": cur_data["interact"]["manipulation"]["hand"],
            "verb": cur_data["interact"]["manipulation"]["verb"],
            "object": cur_data["interact"]["manipulation"]["object"],
            "start": {
                "l": start_waypoint["left"]["3d"],
                "r": start_waypoint["right"]["3d"],
                "l2d": start_waypoint["left"]["2d"],
                "r2d": start_waypoint["right"]["2d"],
            },
            "contact": {
                "time": contact_time,
                "l": (
                    contact_waypoint["left"]["3d"]
                    if contact_waypoint is not None
                    else start_waypoint["left"]["3d"]
                ),
                "r": (
                    contact_waypoint["right"]["3d"]
                    if contact_waypoint is not None
                    else start_waypoint["right"]["3d"]
                ),
                "l2d": (
                    contact_waypoint["left"]["2d"]
                    if contact_waypoint is not None
                    else start_waypoint["left"]["2d"]
                ),
                "r2d": (
                    contact_waypoint["right"]["2d"]
                    if contact_waypoint is not None
                    else start_waypoint["right"]["2d"]
                ),
                "trajectory shape": cur_data["interact"]["approach"]["trajectory"][
                    "shape"
                ],
            },
            "end": {
                "time": end_time,
                "l": end_waypoint["left"]["3d"],
                "r": end_waypoint["right"]["3d"],
                "l2d": end_waypoint["left"]["2d"],
                "r2d": end_waypoint["right"]["2d"],
                "trajectory shape": cur_data["interact"]["manipulation"]["trajectory"][
                    "shape"
                ],
            },
        }
    else:
        last_frame_traj = {
            "hand": cur_data["interact"]["manipulation"]["hand"],
            "verb": cur_data["interact"]["manipulation"]["verb"],
            "object": cur_data["interact"]["manipulation"]["object"],
            "start": {
                "l": start_waypoint["left"]["3d"],
                "r": start_waypoint["right"]["3d"],
                "l2d": start_waypoint["left"]["2d"],
                "r2d": start_waypoint["right"]["2d"],
            },
            "contact": {
                "l": start_waypoint["left"]["3d"],
                "r": start_waypoint["right"]["3d"],
                "l2d": start_waypoint["left"]["2d"],
                "r2d": start_waypoint["right"]["2d"],
            },
            "end": {
                "time": end_time,
                "l": end_waypoint["left"]["3d"],
                "r": end_waypoint["right"]["3d"],
                "l2d": end_waypoint["left"]["2d"],
                "r2d": end_waypoint["right"]["2d"],
                "trajectory shape": cur_data["interact"]["manipulation"]["trajectory"][
                    "shape"
                ],
            },
        }

    # Generate numeric QA dataset (for pretraining)
    numeric_qa_list = generate_pretrain_numeric_QA(
        last_frame_traj,
        cur_data,
    )
    pretrain_numeric_qa_list = []
    for anno_i in range(len(numeric_qa_list)):
        item_cur_anno = cur_data.copy()
        item_cur_anno.pop("interact")
        item_cur_anno.pop("pose_data")

        # motion reasoning: add past motion to input
        if random.random() < 0.3:
            numeric_qa_list[anno_i][
                "q"
            ] = f"Given the past wrist motion: {past_motion}. {numeric_qa_list[anno_i]['q']}"
            numeric_qa_list[anno_i]["field"].append("past_motion")
            numeric_qa_list[anno_i]["value"].append(past_motion_vec)
        item_cur_anno["question"] = numeric_qa_list[anno_i]["q"]
        item_cur_anno["answer"] = numeric_qa_list[anno_i]["a"]
        item_cur_anno["field"] = numeric_qa_list[anno_i]["field"]
        item_cur_anno["value"] = numeric_qa_list[anno_i]["value"]
        pretrain_numeric_qa_list.append(item_cur_anno)

    # Generate finetuning dataset (all QA pairs follow full template)
    # Question: f"Given the past wrist motion: {past_motion}. Where will the hands move to {interact['intention_goal']}?<HOI_QUERY>"
    # Answer: <ACT><START><CONTACT><END>
    item_cur_anno = cur_data.copy()
    interact = item_cur_anno.pop("interact")
    item_cur_anno.pop("pose_data")

    item_cur_anno["question"] = (
        f"Given the past wrist motion: {past_motion}. Where will the hands move to {interact['intention_goal']}?<HOI_QUERY>"
    )
    act_phrase = f"{interact['manipulation']['hand']} hand {interact['manipulation']['verb']} {interact['manipulation']['object']}"
    if contact_waypoint is None:
        contact_waypoint = start_waypoint.copy()
        contact_waypoint["time"] = contact_time = 0.0
    item_cur_anno["answer"] = f"<ACT><START><CONTACT><END>"
    item_cur_anno["field"] = [
        "act.phrase",
        "start.timelr",
        "contact.timelr",
        "end.timelr",
        "past_motion",
    ]
    value_vec = [-1000] * 11
    value_vec1 = [-1000] * 11
    value_vec2 = [-1000] * 11
    value_vec[0] = 0.0
    value_vec1[0] = contact_time
    value_vec2[0] = end_time
    value_vec[1:4] = start_waypoint["left"]["3d"]
    value_vec[4:7] = start_waypoint["right"]["3d"]
    if start_waypoint["left"]["2d"] is not None:
        value_vec[7:9] = start_waypoint["left"]["2d"]
    if start_waypoint["right"]["2d"] is not None:
        value_vec[9:] = start_waypoint["right"]["2d"]
    value_vec1[1:4] = contact_waypoint["left"]["3d"]
    value_vec1[4:7] = contact_waypoint["right"]["3d"]
    if contact_waypoint["left"]["2d"] is not None:
        value_vec1[7:9] = contact_waypoint["left"]["2d"]
    if contact_waypoint["right"]["2d"] is not None:
        value_vec1[9:] = contact_waypoint["right"]["2d"]
    value_vec2[1:4] = end_waypoint["left"]["3d"]
    value_vec2[4:7] = end_waypoint["right"]["3d"]
    if end_waypoint["left"]["2d"] is not None:
        value_vec2[7:9] = end_waypoint["left"]["2d"]
    if end_waypoint["right"]["2d"] is not None:
        value_vec2[9:] = end_waypoint["right"]["2d"]
    item_cur_anno["value"] = [
        np.array([value_vec, value_vec1, value_vec2]),
        past_motion_vec,
        act_phrase,
        np.round(np.hstack([left_pos, right_pos]), 3),
        np.round(np.hstack([left_quat, right_quat]), 4),
    ]
    pretrain_numeric_qa_list.append(item_cur_anno)
    finetune_qa_item = item_cur_anno.copy()

    return pretrain_numeric_qa_list, finetune_qa_item
