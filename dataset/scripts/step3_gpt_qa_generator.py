# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Step 3: Non-Numeric QA Generation

Purpose:
    Generate diverse non-numeric question-answer pairs for semantic, spatial, and motion reasoning.
    This script creates QA pairs covering object recognition, spatial relationships, motion patterns,
    and interaction stages for each valid interaction item after filtering in Step 2.

Dependencies:
    - Runs after: Step 2 (valid_interact_filter.py) - requires filtered interaction annotations

Input:
    - Filtered interaction annotations from Step 2 containing:
        * Intention goals
        * Interaction stages (approach, manipulation)
        * Object and action descriptions
        * Temporal information

Output:
    - Non-numeric QA pairs covering:
        * Current intention goals
        * Which hand will be used
        * What action will occur
        * What object will be manipulated
        * Hand trajectory descriptions
        * Interaction stage information
        * Reasoning about why actions occur

Usage:
    See the generate_qa_pairs() function below for the main entry point.
    This function should be called for each valid interaction annotation.
"""

import json
import time



def generate_qa_pairs(interact_data, llm):
    """Generate non numeric QA pairs for predicting future hand-object interactions
    interact_data (dict): one data annotation sample after filtering
    llm: the llm client (follow step1 to set up your llm client)

    Usage Example:
        # cur_data: one data annotation sample after filtering
        qa_result = generate_qa_pairs(cur_data["interact"], llm)
        if qa_result["valid"] == "valid":
            cur_data["qa_pairs"] = qa_result["qa_pairs"]
    """

    messages = [
        {
            "role": "system",
            "content": """You are a data annotator. Generate diverse question-answer pairs about the provided hand-object interaction data (which represents the next interaction to predict).

                ### Rules:
                * Answers must be short natural phrases. **ONLY use information from the provided data - do not fabricate details like timing.**
                * **Do not use parentheses in questions or answers.**
                * Generate 8-12 QA pairs covering:
                1. **One question asking about the current intention goal** (e.g., "What is the current intention goal?")
                2. **For other questions, inject the intention goal in diverse formats**:
                    - "Given the intention to [goal], xxx"
                    - "To achieve [goal], xxx?"
                    - "While pursuing [goal], xxx?"
                    - "In order to [goal], xxx?"
                    - "When attempting to [goal], xxx?"
                    - "For the purpose of [goal], xxx?"
                    - "As part of [goal], xxx?"
                    - "xxx to accomplish [goal]?"
                3. Which hand will be used next
                4. What action will occur next
                5. What object will be manipulated next
                6. What trajectory the hands will follow next for manipulation
                7. Where the next manipulation interaction will start/end or start and end
                8. What is the next atomic motion (answer should be the atomic description of the next interaction)
                9. Why the next action will happen (answer should be the reasoning)
                10. **If the data contains approach stage before manipulation**: When will the hand approach the object (answer should be the end time of the approach stage)? where the approach will start/end or start and end? what the trajectory be like?
                * Output as JSON array: `[{"q": "...", "a": "..."}]`""",
        },
        {
            "role": "user",
            "content": f"""
                ### Interaction JSON for next interaction:

                {json.dumps(interact_data, indent=2)}

                ### Output:""",
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

            result = json.loads(response_content)

            # Validate it's a list of QA pairs
            if isinstance(result, list) and all(
                "q" in item and "a" in item for item in result
            ):
                return {"valid": "valid", "qa_pairs": result}
            else:
                raise ValueError("Response is not a valid list of QA pairs")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to get valid response after {max_retries} attempts")
                return {
                    "valid": "invalid",
                    "qa_pairs": [],
                }
            else:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(1)
