import json
import os

def process_file(file_path, task_type):
    print(f"Processing {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if not isinstance(data, list):
            print(f"Error: content of {file_path} is not a list.")
            return

        count = 0
        for item in data:
            item['type'] = task_type
            count += 1
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"Success: Added 'type': {task_type} to {count} items in {os.path.basename(file_path)}")

    except Exception as e:
        print(f"Exception while processing {file_path}: {e}")

def main():
    base_dir = "/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/annotations"
    
    # Mapping of filename to task type
    # 1: ScanRefer (Localization)
    # 2: ScanQA (QA)
    tasks = {
        # "scanrefer_mask3d_train.json": 1,
        # "scanqa_train.json": 2,
        "multi3dref_mask3d_train.json": 3,
        "scan2cap_mask3d_train.json": 4,
        "sqa3d_train.json": 5,
    }
    
    for filename, task_type in tasks.items():
        file_path = os.path.join(base_dir, filename)
        process_file(file_path, task_type)

if __name__ == "__main__":
    main()