import os
import json
import cv2
import numpy as np


def find_bounding_boxes(mask):
    """
    从二值掩码图中找到所有目标的边界框。
    此函数与之前完全相同，用于寻找图中值为255的区域。

    参数:
        mask (numpy.array): 一个二维数组，目标像素为255，背景为0。

    返回:
        list: 包含所有边界框的列表，格式为 [x, y, w, h]。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, w, h])
    return bboxes


def create_coco_dataset_sirst(label_dir, file_list_path, output_path, name_suffix):
    """
    根据SIRST数据集的文件列表和命名规则，创建并保存一个COCO格式的JSON数据集。
    此函数已更新，以处理 "__pixels0.png" 的文件名后缀。

    参数:
        label_dir (str): 包含标签掩码图像的文件夹路径。
        file_list_path (str): .txt文件路径，每行列出一个不带扩展名的图像文件名。
        output_path (str): 生成的.json文件的保存路径。
        name_suffix (str): 附加到基础文件名后以定位掩码文件的后缀。
    """
    # 1. 初始化COCO数据结构
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "target", "supercategory": "target"}]
    }

    # 2. 读取图像文件名列表
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            base_filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：无法找到文件列表 '{file_list_path}'。请仔细检查路径。")
        return

    print(f"在 '{file_list_path}' 中找到 {len(base_filenames)} 个文件记录。")

    annotation_id = 0
    # 3. 遍历列表中的每一个基础文件名
    for base_name in base_filenames:
        # 根据新的命名规则构建掩码文件的完整名称
        mask_filename = f"{base_name}{name_suffix}"
        mask_path = os.path.join(label_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"警告：跳过不存在的掩码文件：{mask_path}")
            continue

        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"警告：无法读取图像或格式错误：{mask_path}")
            continue

        height, width = mask_image.shape
        # 使用基础文件名作为 image_id
        image_id = base_name

        # 4. 添加图像信息
        # 假设原始图像是.png格式。如果原始图像是.jpg，请将下面的 ".png" 改为 ".jpg"
        coco_output["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{base_name}.png"
        })

        # 5. 找到边界框
        bboxes = find_bounding_boxes(mask_image)

        # 6. 为每个边界框创建标注信息
        for bbox in bboxes:
            x, y, w, h = bbox
            area = w * h

            # 您提供的示例中有一个'score'字段，这在标准的COCO标注中不常见
            # 这里我们按照标准COCO格式生成，不包含score字段
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x, y, w, h],
                "area": float(area),
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1

    # 7. 将数据写入JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(coco_output, json_file, indent=4)
    except IOError as e:
        print(f"错误：无法写入文件 '{output_path}': {e}")
        return

    print("-" * 30)
    print(f"成功创建COCO数据集：'{output_path}'")
    print(f" -> 总计图像数：{len(coco_output['images'])}")
    print(f" -> 总计标注数：{len(coco_output['annotations'])}")
    print("-" * 30)


if __name__ == "__main__":
    # --- 请根据您的实际文件结构修改以下路径 ---

    # 包含SIRST数据集标签掩码图像的文件夹路径
    SIRST_MASK_DIR = r"E:\DATASET\ITT\instance_format\SIRST\sirst-master\masks\masks"
    # SIRST_MASK_DIR = r"E:\DATASET\ITT\open-sirst-v2\annotations\masks"

    # train.json 的文件名列表
    TRAIN_FILE_LIST = r"E:\DATASET\ITT\instance_format\SIRST\sirst-master\idx_427\trainvaltest.txt"

    # val.json 的文件名列表
    VAL_FILE_LIST = r"E:\DATASET\ITT\instance_format\SIRST\sirst-master\idx_427\trainval.txt"

    # SIRST数据集中，掩码文件名使用的特殊后缀
    FILENAME_SUFFIX = "_pixels0.png"

    # --- 脚本执行 ---
    print("开始处理SIRST数据集...")

    print("\n[阶段 1/2] 正在生成 train.json...")
    create_coco_dataset_sirst(
        label_dir=SIRST_MASK_DIR,
        file_list_path=TRAIN_FILE_LIST,
        output_path=r"E:\DATASET\ITT\instance_format\SIRST\train.json",
        name_suffix=FILENAME_SUFFIX
    )

    print("\n[阶段 2/2] 正在生成 val.json...")
    create_coco_dataset_sirst(
        label_dir=SIRST_MASK_DIR,
        file_list_path=VAL_FILE_LIST,
        output_path=r"E:\DATASET\ITT\instance_format\SIRST\val.json",
        name_suffix=FILENAME_SUFFIX
    )

    print("\n所有任务已完成！")