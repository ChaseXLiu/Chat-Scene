import json
import os
from pathlib import Path
import re
from tqdm import tqdm

def preprocess_multi3drefer_eval_data(input_file: str, output_file: str):
    with open(input_file, 'r') as f:
        data = json.load(f)

    processed_data = []
    for item in tqdm(data, desc="Processing Multi3DRefer evaluation data", unit="item"):
        scene_id = item.get('scene_id', 'unknown')
        obj_id = item.get('obj_id', 0)
        original_prompt = item.get('prompt', '')
        ref_captions = item.get('ref_captions', [])
        eval_type = item.get('eval_type', 'unknown')

        description = ""
        if original_prompt:
            quote_match = re.search(r'"([^"]*)"', original_prompt)
            if quote_match:
                description = quote_match.group(1)
            else:
                description = original_prompt
                print(f"Warning: Could not parse description from prompt: {original_prompt}")

        new_prompt = (
            f"Does anything fit the description of \"{description}\"? "
            "List matching object IDs. Think step by step: "
            "(1) Check object type. (2) Verify spatial relations. (3) List IDs. "
            "Provide only the final answer in the format 'Yes. <OBJXXX>.' or 'No.' unless reasoning is requested."
        )

        new_item = {
            'scene_id': scene_id,
            'obj_id': obj_id,
            'prompt': new_prompt,
            'ref_captions': ref_captions,
            'eval_type': eval_type
        }
        processed_data.append(new_item)

    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

def main():
    multi3drefer_eval_input = '/home/lcx/chat-scene/Chat-Scene/annotations/multi3dref_mask3d_val.json'
    multi3drefer_eval_output = '/home/lcx/chat-scene/Chat-Scene/annotations/multi3drefer_eval_processed_prompt_only.json'
    if os.path.exists(multi3drefer_eval_input):
        preprocess_multi3drefer_eval_data(multi3drefer_eval_input, multi3drefer_eval_output)
        print(f"Processed Multi3DRefer evaluation data saved to {multi3drefer_eval_output}")
    else:
        print(f"Multi3DRefer evaluation input file {multi3drefer_eval_input} not found")

if __name__ == '__main__':
    main()