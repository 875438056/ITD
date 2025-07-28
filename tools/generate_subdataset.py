import json
import os


def create_coco_subset(original_json_path, output_json_path, percentage_to_keep):
    """
    读取一个COCO格式的JSON文件，筛选出指定百分比的图像及其对应的标注，
    并生成一个新的JSON文件。

    参数:
    original_json_path (str): 原始 train.json 文件的路径。
    output_json_path (str): 新生成的子集JSON文件的保存路径。
    percentage_to_keep (float): 要保留的数据百分比 (例如 0.2 表示 20%)。
    """
    print(f"正在读取原始文件: {original_json_path}")

    # 检查原始文件是否存在
    if not os.path.exists(original_json_path):
        print(f"错误: 找不到文件 '{original_json_path}'。请确保该文件与脚本在同一目录下，或者路径正确。")
        return

    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 获取所有图像和标注信息
    images = data['images']
    annotations = data['annotations']

    num_original_images = len(images)
    print(f"原始文件中共有 {num_original_images} 张图像。")

    # 2. 计算需要保留的图像数量并进行筛选
    num_images_to_keep = int(num_original_images * percentage_to_keep)
    images_subset = images[:num_images_to_keep]
    print(f"根据 {percentage_to_keep * 100:.0f}% 的比例，将筛选出 {num_images_to_keep} 张图像。")

    # 3. 获取被选中图像的ID集合，用于快速查找
    image_ids_subset = {img['id'] for img in images_subset}

    # 4. 根据图像ID筛选对应的标注（数据对齐的关键步骤）
    annotations_subset = [anno for anno in annotations if anno['image_id'] in image_ids_subset]
    print(f"成功筛选出 {len(annotations_subset)} 条与所选图像对应的标注。")

    # 5. 构建新的JSON数据结构
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': images_subset,
        'annotations': annotations_subset
    }

    # 6. 保存为新的JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)  # indent=4 让JSON文件格式更美观，易于阅读

    print(f"成功！已生成新的训练子集文件: {output_json_path}")


if __name__ == '__main__':
    # --- 配置信息 ---
    # 您的原始文件名
    input_file = r'E:\DATASET\LJB\LJB_train_coco_jpg_latest_trainset_14015\annotations\val.json'
    # 您希望生成的新文件名
    output_file = r'E:\DATASET\LJB\LJB_train_coco_jpg_latest_trainset_2803(14015)\annotations\val.json'
    # 要筛选的数据比例
    percentage = 0.20

    create_coco_subset(input_file, output_file, percentage)