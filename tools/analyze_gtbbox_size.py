import json
from collections import defaultdict


def format_dim(dim):
    """辅助函数，如果尺寸是整数，则不显示小数位。"""
    if dim == int(dim):
        return str(int(dim))
    else:
        # 对于非整数，保留两位小数
        return f"{dim:.2f}"


def analyze_bbox_hw(ann_file_path, verbose=True):
    """
    分析COCO格式标注文件中每个边界框的(高, 宽)组合、像素面积、数量及其占比。

    参数:
        ann_file_path: COCO标注文件路径
        verbose: 是否打印详细分析结果
    """
    # --- 文件读取与错误处理 ---
    try:
        with open(ann_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 无法找到文件 {ann_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 {ann_file_path} 不是有效的JSON格式。")
        return None

    if 'annotations' not in data:
        raise ValueError("标注文件中找不到 'annotations' 键")

    # --- 数据统计 ---
    # 使用defaultdict来统计每个 (height, width) 组合的数量
    hw_counts = defaultdict(int)
    total_boxes = 0

    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            continue

        if 'bbox' in ann and len(ann['bbox']) == 4:
            width = ann['bbox'][2]
            height = ann['bbox'][3]

            # 使用 (height, width) 元组作为键进行统计
            hw_counts[(height, width)] += 1
            total_boxes += 1

    if total_boxes == 0:
        print("警告: 没有找到有效的边界框标注")
        return defaultdict(int)

    # --- 结果处理与输出 ---
    results = []
    for (h, w), count in hw_counts.items():
        area = h * w
        results.append((area, h, w, count))

    # 按面积大小对结果进行升序排序
    sorted_results = sorted(results, key=lambda item: item[0])

    if verbose:
        print("\n--- 边界框尺寸(高x宽)分析结果 ---")
        print(f"总目标框数量: {total_boxes}")
        print(f"独立尺寸(高x宽)种类数量: {len(hw_counts)}")
        # 增加了横线的长度以适应新的列
        print("-" * 55)
        # 更新了表头
        print("h * w (面积)             ->  数量      (占比)")
        print("-" * 55)

        for area, h, w, count in sorted_results:
            # 【新增】计算数量占比
            percentage = (count / total_boxes) * 100

            h_str = format_dim(h)
            w_str = format_dim(w)

            output_str = f"{h_str}*{w_str}({area:.2f})"

            # 【修改】更新打印格式，加入格式化的百分比
            print(f"{output_str:<25} -> {count:>6} 个 ({percentage:5.2f}%)")

        print("-" * 55)

    return hw_counts


# --- 使用示例 ---
if __name__ == "__main__":
    # 请将这里替换为您的验证集标注文件路径
    # ann_file_path = r'E:\DATASET\LJB\LJB_train_coco_jpg_latest_trainset_14015\annotations\val.json'
    # ann_file_path = r"E:\DATASET\ITT\bbox_format\SIRST\annotations\train.json"
    ann_file_path = r"E:\DATASET\ITT\bbox_format\IRSTD-1k\annotations\train.json"
    # 调用分析函数
    hw_counts = analyze_bbox_hw(ann_file_path)

    if hw_counts:
        print("\n分析完成。")
        # 示例：找到最常见的尺寸组合
        most_common_hw = max(hw_counts, key=hw_counts.get)
        h, w = most_common_hw
        count = hw_counts[most_common_hw]
        print(f"最常见的尺寸是 高 {format_dim(h)} x 宽 {format_dim(w)}, 出现了 {count} 次。")