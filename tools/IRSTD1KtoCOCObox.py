import os
import json
import cv2
import numpy as np


def find_bounding_boxes(mask):
    """
    从二值掩码图中找到所有目标的边界框。
    OpenCV的findContours函数可以有效地找到所有连续的白色像素区域。

    参数:
        mask (numpy.array): 一个二维数组，其中目标像素为255，背景为0。

    返回:
        list: 一个包含所有边界框的列表，每个边界框格式为 [x, y, w, h]。
    """
    # cv2.RETR_EXTERNAL 表示只检测最外层的轮廓。
    # cv2.CHAIN_APPROX_SIMPLE 会压缩水平、垂直和对角线段，只留下它们的端点。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        # 对于每个找到的轮廓，计算其最小正向边界矩形。
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, w, h])
    return bboxes


def create_coco_dataset(label_dir, file_list_path, output_path):
    """
    根据给定的文件列表和标签图像，创建并保存一个COCO格式的JSON数据集。

    参数:
        label_dir (str): 包含标签掩码图像的文件夹路径。
        file_list_path (str): 一个.txt文件路径，每行列出一个不带扩展名的图像文件名。
        output_path (str): 生成的.json文件的保存路径。
    """
    # 1. 初始化COCO数据结构
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "target", "supercategory": "target"}]
    }

    # 2. 读取包含图像文件名的文本文件
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            # 使用列表推导式读取所有行，并用strip()移除可能存在的空白符（如换行符）
            image_filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：无法找到文件列表 '{file_list_path}'。请检查路径是否正确。")
        return

    print(f"在 '{file_list_path}' 中找到 {len(image_filenames)} 个文件记录。")

    annotation_id = 0
    # 3. 遍历列表中的每一个文件名
    for filename in image_filenames:
        # 假设掩码图像是.png格式，如果您的图像是其他格式（如.bmp），请修改此处
        mask_filename = f"{filename}.png"
        mask_path = os.path.join(label_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"警告：跳过不存在的掩码文件：{mask_path}")
            continue

        # 使用OpenCV以灰度模式读取掩码图像
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"警告：无法读取或图像格式错误：{mask_path}")
            continue

        height, width = mask_image.shape
        image_id = filename  # 使用文件名（不含扩展名）作为 image_id

        # 4. 添加图像信息到 'images' 列表
        # 您的示例中file_name是.jpg，所以这里也使用.jpg。
        coco_output["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{filename}.png"
        })

        # 5. 从掩码中找到所有目标的边界框
        bboxes = find_bounding_boxes(mask_image)

        # 6. 为每个找到的边界框创建标注信息
        for bbox in bboxes:
            x, y, w, h = bbox
            area = w * h

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # 类别ID，因为只有一个类别'target'，所以固定为0
                "bbox": [x, y, w, h],
                "area": float(area),  # JSON标准建议将面积存为浮点数
                "iscrowd": 0,  # 0表示非拥挤目标
                "segmentation": []  # 对于边界框任务，此项可为空
            })
            annotation_id += 1

    # 7. 将最终的COCO数据结构写入JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            # indent=4 使JSON文件格式化，更易于阅读
            json.dump(coco_output, json_file, indent=4)
    except IOError as e:
        print(f"错误：无法写入文件 '{output_path}': {e}")
        return

    print("-" * 30)
    print(f"成功创建COCO数据集：'{output_path}'")
    print(f" -> 总计图像数：{len(coco_output['images'])}")
    print(f" -> 总计标注数（边界框）：{len(coco_output['annotations'])}")
    print("-" * 30)


if __name__ == "__main__":
    # --- 请根据您的实际文件结构修改以下路径 ---
    # 使用 r"..." 原始字符串格式可以避免反斜杠带来的转义问题

    # 包含标签掩码图像的文件夹路径 (E:\DATASET\ITT\IRSTD-1k\IRSTD1k_Label)
    LABEL_IMAGE_DIRECTORY = r"E:\DATASET\ITT\IRSTD-1k\IRSTD1k_Label"

    # 包含训练集文件名的文本文件路径 (E:\DATASET\ITT\IRSTD-1k\trainvaltest.txt)
    TRAIN_FILE_LIST = r"E:\DATASET\ITT\IRSTD-1k\trainvaltest.txt"

    # 包含验证/测试集文件名的文本文件路径 (E:\DATASET\ITT\IRSTD-1k\test.txt)
    VAL_FILE_LIST = r"E:\DATASET\ITT\IRSTD-1k\test.txt"

    # --- 脚本执行 ---
    print("开始处理数据集...")

    print("\n[阶段 1/2] 正在生成 train.json...")
    create_coco_dataset(
        label_dir=LABEL_IMAGE_DIRECTORY,
        file_list_path=TRAIN_FILE_LIST,
        output_path=r"E:\DATASET\ITT\IRSTD-1k\train.json"
    )

    print("\n[阶段 2/2] 正在生成 val.json...")
    create_coco_dataset(
        label_dir=LABEL_IMAGE_DIRECTORY,
        file_list_path=VAL_FILE_LIST,
        output_path=r"E:\DATASET\ITT\IRSTD-1k\val.json"
    )

    print("\n所有任务已完成！")