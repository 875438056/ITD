import json

# 替换成您 train.json 的实际路径
annotation_file = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/annotations/train.json'

print(f"Loading annotation file: {annotation_file}")

with open(annotation_file, 'r') as f:
    data = json.load(f)

# 1. 从 "categories" 中获取所有合法的 category ID
valid_category_ids = set()
if 'categories' in data and data['categories']:
    for category in data['categories']:
        valid_category_ids.add(category['id'])
else:
    print("Error: 'categories' key is missing or empty in the annotation file!")
    exit()

print(f"Found valid category IDs: {valid_category_ids}")

# 2. 遍历所有标注，检查它们的 category_id
error_found = False
for annotation in data['annotations']:
    anno_id = annotation['id']
    category_id = annotation['category_id']

    if category_id not in valid_category_ids:
        print(f"--> Error found! Annotation ID: {anno_id} has an invalid category_id: {category_id}")
        error_found = True

if not error_found:
    print("\nValidation finished. No invalid category_ids found. Your train.json seems correct.")
else:
    print("\nPlease correct the invalid annotations listed above in your dataset.")