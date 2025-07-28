# ITD
Infrared Tiny Target Detection  
we will upload related code after paper is accepted.

# Datasets sources
[SIRST](https://github.com/YimianDai/sirst)
[IRSTD-1k](https://github.com/RuiZhang97/ISNet)
[ITT-2_15-original](https://drive.google.com/drive/folders/166iNTmKyahH7TPzSQjt5-1j4BEX9uw-Z?usp=drive_link)
[ITT-2_15-annotations](https://drive.google.com/drive/folders/166iNTmKyahH7TPzSQjt5-1j4BEX9uw-Z?usp=drive_link)

# Prerequisites
-mmcv	2.1.0  
-mmdet	3.3.0  
-mmengine	0.10.7   
-numpy	1.26.4  
-opencv-python	4.11.0.86  
-torch	2.6.0+cu126  
-torchvision	0.21.0+cu126

# Det Results
| Models         | mAP   | mAP50 | Recall | FLOPs   | Params  | FPS  |
|----------------|-------|-------|--------|---------|---------|------|
| Faster RCNN [15] | 0.413 | 0.543 | 0.470  | 134 G   | 41.0 M  | 20   |
| YOLOv5s [5]    | 0.437 | 0.644 | 0.543  | 15.9 G  | 7.02 M  | 185.1|
| FCOS [16]      | 0.006 | 0.010 | 0.003  | 31.1 G  | 7.90 M  | 5.6  |
| CenterNet [36] | 0.238 | 0.422 | 0.311  | 0.59 G  | 1.09 M  | 32.4 |
| L-FFCA-YOLO [11] | 0.440 | 0.633 | 0.538 | 37.6 G  | 5.05 M  | 90.9 |
| YOLOv5s-S2 [38] | 0.605 | 0.745 | 0.672 | 22.0 G  | 7.23 M  | 129.8|
| EFLNet [9]     | 0.205 | 0.349 | 0.306  | 8.24 G  | 33.0 M  | 12.5 |
| **Ours**       | **0.817** | **0.868** | **0.859** | **2.04 G** | **1.17 M** | **30.8** |
