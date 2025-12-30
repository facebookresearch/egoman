# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 6: Trajectory Quality Filtering (GPT-Based Validation)

Purpose:
    Apply GPT-powered validation to filter out trajectories where the interaction
    phrase doesn't match the visual scene. This is an optional second-stage filter after
    rule-based filtering (step6_traj_quality_filter_rules.py) for stricter quality control.

Dependencies:
    - Runs after: Rule-based filtering (step6_traj_quality_filter_rules.py) - recommended
    - Runs after: Step 5 (reason_numeric_qa_generator.py) - requires trajectory data
    - Runs before: Final dataset compilation

Input:
    - Trajectory data from Step 5 (or after rule-based filtering) containing:
        * image_path: Path to reference frame image
        * phrase_str: Action phrase describing the interaction
        * Example: "left hand open refrigerator door"

Output:
    - Boolean validation result with optional reason:
        * {"valid": True} - Interaction matches visual scene
        * {"valid": False, "reason": "<explanation>"} - Validation failed

Validation Criteria (ALL must be true):
    1. **Realistic Interaction:**
       - Phrase describes a plausible hand-object interaction
       - Relevant to what can be observed in the image

    2. **Object Visibility:**
       - Target object is clearly visible in the image
       - Note: Hand visibility is NOT required (hand may be out of frame)

    3. **Image Quality:**
       - Image allows clear identification of the target object
       - Not blurred, occluded, or otherwise degraded

    4. **Unambiguous Target:**
       - Instruction clearly specifies which object to interact with
       - No confusion about which object is the target

Example Failure Reasons:
    - "object not visible" - Target object cannot be seen in image
    - "ambiguous target" - Multiple similar objects, unclear which to interact with
    - "poor image quality" - Image too blurred/dark to identify objects clearly
    - "unrealistic interaction" - Phrase describes implausible action

GPT Prompt Strategy:
    - Uses GPT with base64-encoded images
    - Returns structured JSON for reliable parsing
    - Retry logic (3 attempts) for robustness

Usage:
    result = trajectory_gpt_filter(
        image_path="path/to/frame.jpg",
        phrase_str="left hand grasp cup",
        llm=gpt_client,
        max_retries=3
    )

    if result["valid"]:
        # Keep trajectory
        valid_trajectories.append(cur_data)
    else:
        # Discard with reason
        print(f"Rejected: {result['reason']}")

Integration Example:
    # After rule-based filtering
    rule_filtered = trajectory_quality_filtering_by_rules(all_data)

    # Apply GPT semantic validation
    final_valid = []
    for item in tqdm(rule_filtered):
        result = trajectory_gpt_filter(
            item['image_path'],
            item['action_phrase'],
            llm_client
        )
        if result['valid']:
            final_valid.append(item)

Note: This filter is computationally expensive (requires GPT calls).
      Use sparingly and consider batching for efficiency.
      Configure your GPT API credentials before running (follow step1 setup).
"""

import base64
import json
import logging
import os
import re
import time

from PIL import Image


def encode_image(image_path):
    """Encode image to base64 for GPT-4 vision processing"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def trajectory_gpt_filter(image_path, phrase_str, llm, max_retries=3):
    """
    GPT filter that validates if phrase_str interaction is relevant to the image.

    Args:
        image_path: Path to the image file
        phrase_str: Phrase describing the interaction
        llm: the llm client (follow step1 to set up your llm client)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        dict: JSON with "valid" key (True/False) and optional "reason" for validation
    """

    prompt = f"""Analyze this image with the interaction phrase: "{phrase_str}"

        Validation criteria (ALL must be true):
        1. Phrase describes a realistic hand-object interaction relevant to the image
        2. Target object is clearly visible in the image (hand visibility not required)
        3. Image quality allows clear identification of the target object
        4. Instruction unambiguously specifies which object to interact with

        Return ONLY valid JSON:
        {{"valid": true}} - all criteria met
        {{"valid": false, "reason": "which criterion failed"}} - any criterion fails

        Example reasons: "object not visible", "ambiguous target", "poor image quality", "unrealistic interaction"
        """

    for attempt in range(max_retries):
        try:
            content = [{"type": "text", "text": prompt}]

            if image_path and os.path.exists(image_path):
                try:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            },
                        }
                    )
                except Exception as e:
                    logging.error(f"Error encoding image {image_path}: {e}")
                    return {"valid": False, "reason": f"Image encoding error: {e}"}
            else:
                return {"valid": False, "reason": "Image path does not exist"}

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert evaluator for hand-object interaction validation. Always respond with valid JSON only.",
                },
                {"role": "user", "content": content},
            ]

            response = llm.invoke(messages)
            result_text = response.content.strip()

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)

            result_json = json.loads(result_text)

            if "valid" in result_json:
                logging.info(
                    f"Successfully validated on attempt {attempt + 1}: {result_json}"
                )
                return result_json
            else:
                logging.warning(
                    f"Response missing 'valid' key on attempt {attempt + 1}"
                )

        except json.JSONDecodeError as e:
            logging.warning(
                f"JSON parsing error on attempt {attempt + 1}/{max_retries}: {e}"
            )
            if attempt == max_retries - 1:
                return {
                    "valid": False,
                    "reason": f"Failed to parse GPT response after {max_retries} attempts",
                }
            time.sleep(1)

        except Exception as e:
            logging.error(
                f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}"
            )
            if attempt == max_retries - 1:
                return {"valid": False, "reason": f"Error: {str(e)}"}
            time.sleep(1)

    return {"valid": False, "reason": f"Failed after {max_retries} attempts"}
