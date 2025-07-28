import os
import json
import cv2
from collections import defaultdict


def draw_gt_boxes_on_images_separated(base_path, annotation_files):
    """
    根据COCO格式的JSON标注文件，在图片上绘制真实边界框(GT boxes)并保存。
    为每个JSON文件创建独立的输出文件夹。

    参数:
    base_path (str): 数据集的根路径。
    annotation_files (list): 需要处理的JSON文件名列表 (例如 ['train.json', 'val.json'])。
    """
    # 1. 定义输入路径
    image_folder = os.path.join(base_path, 'images')
    annotation_folder = os.path.join(base_path, 'annotations')

    # 2. 遍历每个JSON标注文件 (train.json, val.json)
    for ann_file in annotation_files:
        json_path = os.path.join(annotation_folder, ann_file)

        if not os.path.exists(json_path):
            print(f"警告: 找不到标注文件 {json_path}，已跳过。")
            continue

        # --- 修改点: 动态生成并创建输出文件夹 ---
        # 从 "train.json" 生成 "train"
        file_base_name = os.path.splitext(ann_file)[0]
        # 生成 "train_images_with_GT_boxes"
        output_folder_name = f"{file_base_name}_images_with_GT_boxes"
        output_folder = os.path.join(base_path, output_folder_name)

        # 检查并创建对应的输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"已创建输出文件夹: {output_folder}")
        # --- 修改结束 ---

        print(f"--- 正在处理文件: {ann_file} ---")

        # 读取JSON文件内容
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 3. 创建高效的数据结构以便查找
        image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}
        image_id_to_annotations = defaultdict(list)
        for ann in data['annotations']:
            image_id_to_annotations[ann['image_id']].append(ann['bbox'])

        # 4. 遍历有标注的图片，绘制边界框并保存
        total_images = len(image_id_to_annotations)
        if total_images == 0:
            print(f"在 {ann_file} 中没有找到任何带标注的图片。")
            continue

        print(f"在 {ann_file} 中找到 {total_images} 张带有标注的图片，将保存到 {output_folder_name}")

        for i, (image_id, bboxes) in enumerate(image_id_to_annotations.items()):
            # file_name = image_id_to_filename.get(image_id) + '.PNG'
            file_name = image_id_to_filename.get(image_id)
            if not file_name:
                print(f"警告: 在 'images' 列表中找不到 image_id '{image_id}' 对应的文件名。")
                continue

            image_path = os.path.join(image_folder, file_name)
            if not os.path.exists(image_path):
                print(f"警告: 找不到图片文件 {image_path}，已跳过。")
                continue

            image = cv2.imread(image_path)

            for bbox in bboxes:
                x, y, w, h = [int(coord) for coord in bbox]
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                color = (0, 255, 0)
                thickness = 1
                cv2.rectangle(image, top_left, bottom_right, color, thickness)

            # 构建输出文件路径，现在它指向特定于该json文件的文件夹
            output_path = os.path.join(output_folder, file_name)

            cv2.imwrite(output_path, image)

            if (i + 1) % 100 == 0 or (i + 1) == total_images:
                print(f"进度: {i + 1}/{total_images} | 已保存到: {output_path}")

    print("\n--- 所有处理完成！ ---")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 请根据您的实际情况修改这里的路径

    # 1. 数据集根目录
    # dataset_base_path = r'E:\DATASET\LJB\LJB_train_coco_jpg_latest_trainset_14015'
    # dataset_base_path = r'E:\DATASET\LJB\LJB_train_coco_jpg_1&2test_dataset_27000-416'
    # dataset_base_path = r'E:\DATASET\SIRST\SIRST_COCO'
    # dataset_base_path = r'E:\DATASET\ITT\bbox_format\IRSTD-1k'
    # dataset_base_path = r'E:\DATASET\ITT\bbox_format\SIRST'
    dataset_base_path = r'E:\DATASET\ITT\bbox_format\SIRST-V2'

    # 2. 需要处理的JSON文件名
    annotations_to_process = ['train.json', 'val.json']

    # 执行函数
    draw_gt_boxes_on_images_separated(dataset_base_path, annotations_to_process)