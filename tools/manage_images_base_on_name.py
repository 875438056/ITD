import os
import shutil


def copy_images_to_organized_folders(base_path, source_folder_name):
    """
    根据文件名将图片复制并分类整理到新的子目录中。
    原始文件将保持不变。
    文件名格式: part1_part2_part3.jpg
    复制规则: file.jpg -> /part1/part2/file.jpg

    参数:
    base_path (str): 数据集的根路径，新的分类文件夹将创建在这里。
    source_folder_name (str): 存放原始图片的文件夹名称。
    """
    # 1. 定义源文件夹路径
    source_path = os.path.join(base_path, source_folder_name)

    if not os.path.isdir(source_path):
        print(f"错误：源文件夹 '{source_path}' 不存在。请检查路径。")
        return

    print(f"开始扫描源文件夹: {source_path}")

    # 2. 获取源文件夹下所有文件的列表
    files_to_process = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    total_files = len(files_to_process)
    print(f"找到 {total_files} 个文件需要复制。")

    # 3. 遍历每个文件
    for i, filename in enumerate(files_to_process):
        # 从文件名中分离出主干名和扩展名，例如 '1_11_423' 和 '.jpg'
        name_part, extension = os.path.splitext(filename)

        # 4. 解析文件名
        try:
            # 按下划线分割文件名
            parts = name_part.split('_')
            if len(parts) < 2:
                # 如果文件名不符合 "part1_part2_..." 格式，则跳过
                print(f"警告: 文件名 '{filename}' 不符合格式，已跳过。")
                continue

            dir1 = parts[0]
            dir2 = parts[1]

            # 5. 构建目标文件夹路径
            # 目标文件夹将位于 base_path 下，例如 E:\...\1\11
            destination_dir = os.path.join(base_path, dir1, dir2)

            # 6. 如果目标文件夹不存在，则创建它
            os.makedirs(destination_dir, exist_ok=True)

            # 7. 构建文件的完整源路径和目标路径
            current_file_path = os.path.join(source_path, filename)
            destination_file_path = os.path.join(destination_dir, filename)

            # 8. 复制文件 (这是核心修改点)
            shutil.copy(current_file_path, destination_file_path)

            # 打印进度
            print(f"进度 {i + 1}/{total_files} | 已复制: {filename} -> {destination_file_path}")

        except Exception as e:
            print(f"处理文件 '{filename}' 时发生错误: {e}")

    print("\n--- 所有文件复制完成！ ---")
    print(f"原始的 '{source_path}' 文件夹中的文件保持不变。")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 请确认以下路径是否正确

    # 1. 数据集根目录。新的分类文件夹将创建在这里。
    # 注意路径前的 'r'，它可以防止反斜杠引起转义问题
    # dataset_base_path = r'E:\DATASET\LJB\LJB_train_coco_jpg_latest_trainset_14015'
    dataset_base_path = r'E:\DATASET\LJB\LJB_train_coco_jpg_1&2test_dataset_27000-416'
    # 2. 存放原始图片的文件夹名
    images_folder = 'images'

    # 执行整理函数
    copy_images_to_organized_folders(dataset_base_path, images_folder)