# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 2: Valid Interaction Filtering

Purpose:
    Apply rule-based and GPT-powered filters to remove invalid annotations from Step 1.
    This ensures interactions are realistic, properly timed, semantically meaningful, and
    aligned with reference atomic descriptions. Also generates high-level intention summaries.

Dependencies:
    - Runs after: Step 1 (gpt_anno_interact.py) - requires raw interaction annotations
    - Runs before: Step 3 (gpt_qa_generator.py) & Step 4 (6dof_traj_process.py)

Input:
    - Raw interaction annotations from Step 1 containing:
        * Interaction stages (approach, manipulation)
        * Timestamps and trajectories
        * Reference atomic descriptions from source dataset
        * Intention goals

Output:
    - Filtered valid interactions with:
        * Temporal consistency: Start < end times, proper stage alignment
        * Semantic relevance: Aligned with reference atomic descriptions
        * Realism validation: Actual hand-object interactions (not air movement)
        * Duration constraints: 0.25s ≤ duration ≤ 4.5s
        * High-level intention goal summary
        * Relative timestamps normalized to interaction start (t=0)

Filtering Criteria:

1. **Rule-Based Filters:**
   - Temporal validity: end_time > start_time for all stages
   - Stage alignment: approach.end_time ≈ manipulation.start_time (within 0.5s)
   - Approach duration: ≥ 0.25s if present
   - Total interaction duration: 0.25s ≤ duration ≤ 4.5s
   - Two-stage requirement: Valid approach must connect to manipulation

2. **GPT-Based Semantic Filters:**
   - Relevance check: Interaction aligns with reference atomic descriptions
     (Note: Not overly strict - interaction can be any sub-stage of reference sequence)
   - Reality check: Represents actual hand-object interaction by subject
     (Not: air movement, other participants, body movement, looking, walking)
   - Intention goal extraction: Summarize high-level objective

3. **Post-Processing:**
   - Normalize timestamps: All times relative to interaction start (t=0)
   - Snap approach end to manipulation start for consistency
   - Calculate and store start_sec, end_sec for clip extraction

Usage:
    valid_annos = filter_valid_interaction(
        all_data_list=raw_annotations,
        llm=gpt_client,
        valid_output_path="path/to/valid_interactions.pkl"
    )

Output Format:
    Each valid interaction contains:
    - interact: {two_stage, approach, manipulation with normalized timestamps}
    - intention: High-level intention goal
    - start_sec, end_sec: Absolute clip boundaries
    - video, timestamp, dataset: Source metadata
"""

import json
import pickle
import time
from tqdm import tqdm


def filter_valid_interaction_by_gpt_onesample(cur_data, llm):
    """Filter out invalid interactions by GPT
    cur_data (dict): one data annotation sample after filtering by rules
    llm: the llm client (follow step1 to set up your llm client)
    """
    interact_data = cur_data["interact"]
    ref_annos = cur_data["ref_anno"]
    messages = [
        {
            "role": "system",
            "content": """You are an expert in analyzing hand-object interactions. Given an interaction description and reference atomic descriptions, judge:
                1. Whether the interaction is relevant to the reference atomic descriptions. Note: The reference atomic descriptions describe the whole action sequence. If the interaction can be potentially any sub-stage or part of this sequence, it should be considered relevant. Do not be overly strict unless there is obvious contradiction.
                2. Whether this represents a real hand-object interaction of the subject C (not just hand movement in air, not movement of other participants, not body movement, not looking at something, not walking, etc.)
                3. Summarize the high-level intention goal of this interaction

                Return ONLY a valid JSON with these keys:
                {
                    "valid": "valid/invalid",
                    "intention_goal": "high level goal description",
                }
            """,
        },
        {
            "role": "user",
            "content": f"""
                Interaction data: {interact_data["atomic_description"]}

                Reference atomic descriptions: {json.dumps(ref_annos, indent=2)}

                Please analyze and return the JSON judgment.
            """,
        },
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            ai_message = llm.invoke(messages)
            response_content = ai_message.content.strip()

            # Try to parse the JSON response
            if response_content.startswith("```json"):
                response_content = (
                    response_content.strip("```json").strip("```").strip()
                )
            elif response_content.startswith("```"):
                response_content = response_content.strip("```").strip()

            result = eval(response_content)
            if result["valid"] == "invalid":
                print(ref_annos, interact_data["atomic_description"], result)

            # Validate required keys exist
            if all(key in result for key in ["valid", "intention_goal"]):
                return result
            else:
                raise ValueError("Missing required keys in response")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to get valid response after {max_retries} attempts")
                return {
                    "valid": "invalid",
                    "intention_goal": "Unknown",
                }
            else:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(1)



def filter_valid_interaction(all_data_list, llm, valid_output_path=None):
    """
    Filter out invalid interactions
    all_data_list (list of dict): a list of data annotation from step1
    llm: the llm client (follow step1 to set up your llm client)
    valid_output_path (str): the path to save the valid annotation
    """
    valid_all = []
    for cur_data in tqmd(all_data_list):
        new_cur_data = cur_data.copy()
        try:
            ai_anno = eval(new_cur_data["ai_message"])
        except:
            # remove invalid annotation
            continue

        interaction = ai_anno["interactions"]
        new_cur_data["intention"] = ai_anno["intent"]

        # remove empty interactions
        if len(interaction) == 0:
            continue


        for ori_interact in interaction:
            # First filter interaction by rules
            interact = ori_interact.copy()
            new_cur_data["interact"] = {}
            new_cur_data["interact"]["two_stage"] = False
            valid_flag = True
            try:
                if "approach" in interact:
                    # the end time should be larger than the start time
                    if (
                        interact["approach"]["end_time"]
                        < interact["approach"]["start_time"]
                    ):
                        valid_flag = False

                    # the end of approach should be close to the start of manipulation
                    if (
                        abs(
                            float(interact["approach"]["end_time"])
                            - interact["manipulation"]["start_time"]
                        )
                        > 0.5
                        or interact["approach"]["end_time"]
                        - interact["approach"]["start_time"]
                        <= 0.25
                    ):
                        valid_flag = False
                    interact["approach"]["end_time"] = interact["manipulation"][
                        "start_time"
                    ]

                    if "manipulation" not in interact:
                        valid_flag = False
                    else:
                        # the end time should be larger than the start time
                        if (
                            interact["manipulation"]["end_time"]
                            < interact["manipulation"]["start_time"]
                        ):
                            valid_flag = False
                    if valid_flag:
                        new_cur_data["interact"]["two_stage"] = True

                if "manipulation" in interact:
                    # the end time should be larger than the start time
                    if (
                        interact["manipulation"]["end_time"]
                        < interact["manipulation"]["start_time"]
                    ):
                        valid_flag = False
                else:
                    valid_flag = False
            except:
                valid_flag = False

            if not valid_flag:
                continue

            if "approach" in interact:
                very_start_time = interact["approach"]["start_time"]
            else:
                very_start_time = interact["manipulation"]["start_time"]

            # identify the start and end time of the interaction
            very_end_time = interact["manipulation"]["end_time"]
            new_cur_data["start_sec"] = round(very_start_time, 1)
            new_cur_data["end_sec"] = round(very_end_time, 1)

            # remove interactions that are too short or too long
            if (
                very_end_time - very_start_time <= 0.25
                or very_end_time - very_start_time >= 4.5
            ):
                continue
            count += 1

            # calculate relative timestamp of interaction stages
            if "approach" in interact:
                interact["approach"]["start_time"] = 0.0
                interact["approach"]["end_time"] = round(
                    interact["approach"]["end_time"] - very_start_time, 1
                )
            interact["manipulation"]["start_time"] = round(
                interact["manipulation"]["start_time"] - very_start_time,
                1,
            )
            interact["manipulation"]["end_time"] = round(
                interact["manipulation"]["end_time"] - very_start_time, 1
            )
            new_cur_data["interact"].update(interact)

            # second filter interaction by GPT
            validity_judgment = filter_valid_interaction_by_gpt_onesample(
                new_cur_data, llm
            )

            # remove invalid interactions after GPT filtering
            if validity_judgment["valid"] == "invalid":
                continue
            # Add intention goal as a new key to the json
            new_cur_data["intention"] = validity_judgment[
                "intention_goal"
            ]
            new_cur_data["interact"]["intention_goal"] = validity_judgment["intention_goal"]

            valid_all.append(new_cur_data.copy())

    if valid_output_path is not None:
        pickle.dump(valid_all, open(valid_output_path, "wb"))
    return valid_all
