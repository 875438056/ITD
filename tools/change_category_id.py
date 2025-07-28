import json
import os
import shutil
from typing import List

# --- 请在这里配置您需要修改的 JSON 文件路径 ---
# --- 您可以添加任意多个文件路径到这个列表中 ---
ANNOTATION_FILES: List[str] = [
    'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/annotations/train.json',
    'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/annotations/val.json',
]


# ----------------------------------------------------

def update_coco_ids(file_path: str, old_id: int = 1, new_id: int = 0):
    """
    Reads a COCO-style JSON file, backs it up, and updates category and
    annotation IDs from an old value to a new value.

    Args:
        file_path (str): The full path to the COCO annotation JSON file.
        old_id (int): The category ID to be replaced.
        new_id (int): The new category ID.
    """
    print("-" * 50)
    print(f"Processing file: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"--> ERROR: File not found at '{file_path}'. Skipping.")
        return

    # 1. 创建备份
    backup_path = file_path + '.bak'
    try:
        shutil.copy(file_path, backup_path)
        print(f"--> Successfully created backup: {backup_path}")
    except Exception as e:
        print(f"--> ERROR: Could not create backup file. Aborting. Error: {e}")
        return

    # 2. 读取 JSON 文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"--> ERROR: Could not read or parse JSON file. Aborting. Error: {e}")
        return

    # 3. 修改 "categories" 部分
    if 'categories' in data and isinstance(data['categories'], list):
        for category in data['categories']:
            if category.get('id') == old_id:
                category['id'] = new_id
                print(f"--> Updated category '{category.get('name', 'N/A')}' from id {old_id} to {new_id}.")
    else:
        print("--> WARNING: 'categories' key not found or is not a list. Skipping category update.")

    # 4. 修改 "annotations" 部分
    updated_annotations_count = 0
    if 'annotations' in data and isinstance(data['annotations'], list):
        for annotation in data['annotations']:
            if annotation.get('category_id') == old_id:
                annotation['category_id'] = new_id
                updated_annotations_count += 1
    else:
        print("--> WARNING: 'annotations' key not found or is not a list. Skipping annotation update.")

    print(f"--> Found and updated {updated_annotations_count} annotations from category_id {old_id} to {new_id}.")

    # 5. 将修改后的内容写回原文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 使用 indent=4 参数可以保持 JSON 文件的可读性
            json.dump(data, f, indent=4)
        print(f"--> Successfully saved modified data back to {file_path}")
    except Exception as e:
        print(f"--> ERROR: Could not save the modified file. Please check permissions. Error: {e}")
        print(f"--> Your original data is safe in the backup file: {backup_path}")

    print("-" * 50)


if __name__ == "__main__":
    print("Starting COCO category ID update process...")
    for coco_file in ANNOTATION_FILES:
        update_coco_ids(coco_file)
    print("\nProcess finished.")