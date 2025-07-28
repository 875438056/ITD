import cv2
import os
import numpy as np
from tqdm import tqdm


def compute_mean_std(img_dir):
    mean = np.zeros(3)
    std = np.zeros(3)
    img_list = [os.path.join(img_dir, fname)
                for fname in os.listdir(img_dir)
                if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]

    n = len(img_list)
    print(f"共发现 {n} 张图像，开始计算...")

    for img_path in tqdm(img_list):
        img = cv2.imread(img_path).astype(np.float32)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
        img /= 255.0  # 归一化到 [0,1] 再计算均值与标准差（更稳定）

        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))

    mean /= n
    std /= n
    # 如果你在模型里用的是 0~255 的归一化方式，可以乘回来
    mean *= 255
    std *= 255

    return mean.tolist(), std.tolist()


# 示例用法：
img_dir = 'E:/DATASET/LJB/LJB_train_coco_jpg_latest_trainset_14015/images'
mean, std = compute_mean_std(img_dir)
print("mean =", mean)
print("std =", std)
