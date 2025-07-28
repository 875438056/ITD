# models/losses/centernet_kl_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss
import numpy as np
import ot

def centernet_heatmap_kl_loss(pred_heatmap,
                              target_heatmap,
                              weight=None,
                              reduction='mean',
                              avg_factor=None,
                              temperature=1.0,
                              eps=1e-8):
    """Calculate KL Divergence loss for CenterNet heatmaps.

    Args:
        pred_heatmap (torch.Tensor): Predicted heatmap with shape (N, C, H, W).
        target_heatmap (torch.Tensor): Target heatmap with shape (N, C, H, W).
        weight (torch.Tensor, optional): Element-wise weights with shape (N, C, H, W).
        reduction (str): Reduction method ('mean', 'sum', 'none').
        avg_factor (int, optional): Average factor for loss normalization.
        temperature (float): Temperature for softening distributions.
        eps (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: KL divergence loss.
    """
    # Apply temperature scaling
    pred_scaled = pred_heatmap / temperature
    target_scaled = target_heatmap / temperature

    # Apply sigmoid to convert to probabilities
    pred_prob = torch.sigmoid(pred_scaled)
    target_prob = torch.sigmoid(target_scaled)

    # Add small epsilon to avoid log(0)
    pred_prob = torch.clamp(pred_prob, min=eps, max=1 - eps)
    target_prob = torch.clamp(target_prob, min=eps, max=1 - eps)

    # Calculate KL divergence for each pixel
    # KL(target || pred) = target * log(target/pred) + (1-target) * log((1-target)/(1-pred))
    kl_pos = target_prob * torch.log(target_prob / pred_prob)
    kl_neg = (1 - target_prob) * torch.log((1 - target_prob) / (1 - pred_prob))
    kl_loss = kl_pos + kl_neg

    # Apply pixel-wise weights if provided
    if weight is not None:
        kl_loss = kl_loss * weight

    # Reduce loss
    if reduction == 'none':
        return kl_loss
    elif reduction == 'mean':
        if avg_factor is not None:
            return kl_loss.sum() / avg_factor
        else:
            return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def kl_loss(pred_heatmap,
            target_heatmap,
            weight=None,
            temperature=1.0,
            reduction='mean',
            avg_factor=None):
    """Focal KL Loss for CenterNet heatmaps.

    Combines focal loss weighting with KL divergence to handle
    the extreme class imbalance in heatmaps.

    Args:
        pred_heatmap (torch.Tensor): Predicted heatmap (N, C, H, W).
        target_heatmap (torch.Tensor): Target heatmap (N, C, H, W).
        weight (torch.Tensor, optional): Element-wise weights.
        alpha (float): Focal loss alpha parameter for positive samples.
        beta (float): Focal loss beta parameter for negative samples.
        temperature (float): Temperature scaling.
        reduction (str): Reduction method.
        avg_factor (int, optional): Average factor.

    Returns:
        torch.Tensor: Focal KL loss.
    """

    # pred_prob = torch.softmax(pred_heatmap / temperature, dim=1)
    # target_prob = torch.softmax(target_heatmap / temperature, dim=1)

    pred_prob = torch.sigmoid(pred_heatmap / temperature)
    target_prob = torch.sigmoid(target_heatmap / temperature)

    # Clamp probabilities
    eps = 1e-6
    pred_prob = torch.clamp(pred_prob, min=eps, max=1 - eps)
    target_prob = torch.clamp(target_prob, min=eps, max=1 - eps)

    # Calculate KL divergence
    kl_pos = target_prob * torch.log(target_prob / pred_prob)
    kl_neg = (1 - target_prob) * torch.log((1 - target_prob) / (1 - pred_prob))
    kl_loss = kl_pos + kl_neg

    # Apply additional weights
    if weight is not None:
        kl_loss = kl_loss * weight

    # Reduce loss
    return weight_reduce_loss(kl_loss, None, reduction, avg_factor)


import torch.nn.functional as F
def kl_loss1(pred_heatmap,
            target_heatmap,
            weight=None,
            temperature=1.0,
            reduction='mean',
            avg_factor=None):
    """
    修正后的KL Loss

    Args:
        pred_heatmap (torch.Tensor): 预测的热图 logits (N, C, H, W).
        target_heatmap (torch.Tensor): 目标热图，本身就是概率分布 (N, C, H, W).
        ...
    """

    # 1. 对预测值应用 Softmax 得到预测概率分布
    pred_prob = F.log_softmax(pred_heatmap / temperature, dim=1)

    # 2. 目标值 target_heatmap 本身就是目标概率分布，不需要 softmax
    #    注意：需要确保 target_heatmap 的值在 [0, 1] 区间
    target_prob = target_heatmap

    # Clamp probabilities to avoid log(0)
    eps = 1e-6
    pred_prob = torch.clamp(pred_prob, min=eps, max=1 - eps)
    # Target 也可能需要 clamp，特别是当它不是严格的概率分布时
    target_prob = torch.clamp(target_prob, min=eps, max=1 - eps)

    # 3. 使用正确的KL散度公式 (移除了 kl_neg)
    # D_KL(target || pred) = sum(target * log(target / pred))
    kl_loss = F.kl_div(pred_prob, target_prob, reduction='none')

    # 对所有维度求和得到每个样本的loss
    kl_loss = torch.sum(kl_loss, dim=(1, 2, 3))

    # 应用focal loss思想：只在 target > 0 的地方计算损失
    # CenterNet中的Focal Loss思想是只惩罚那些在GT位置预测错误的情况
    # 一个简单的实现是只在 target_heatmap > 0 的地方计算loss
    # 这可以通过乘以一个mask实现，或者直接利用 target_prob 在计算中为0的特性
    # kl_loss = kl_loss * (target_heatmap > 0).float() # 这是一个可选的增强

    if reduction == 'mean':
        if avg_factor is not None:
            loss = kl_loss.sum() / avg_factor
        else:
            loss = kl_loss.mean()
    elif reduction == 'sum':
        loss = kl_loss.sum()
    else:
        loss = kl_loss

    # Apply additional weights
    if weight is not None:
        loss = loss * weight

    # Reduce loss
    return weight_reduce_loss(loss, None, reduction, avg_factor)


@MODELS.register_module()
class KLLoss(nn.Module):
    """KL Divergence Loss for CenterNet Heatmaps.

    This loss is designed specifically for CenterNet-style heatmap regression,
    where the target is a Gaussian heatmap and we want to match the predicted
    distribution to the target distribution.

    Args:
        temperature (float): Temperature for distribution softening. Defaults to 1.0.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 temperature=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted heatmap with shape (N, C, H, W).
            target (torch.Tensor): Target heatmap with shape (N, C, H, W).
            weight (torch.Tensor, optional): Element-wise weights.
            avg_factor (int, optional): Average factor for normalization.
            reduction_override (str, optional): Override reduction method.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        reduction = reduction_override if reduction_override else self.reduction


        loss = kl_loss1(
            pred, target, weight,
            temperature=self.temperature,
            reduction=reduction,
            avg_factor=avg_factor
        )

        return self.loss_weight * loss


def focal_kl_loss(pred_heatmap,
                  target_heatmap,
                  weight=None,
                  alpha=2.0,
                  beta=4.0,
                  temperature=1.0,
                  reduction='mean',
                  avg_factor=None):
    """Focal KL Loss for CenterNet heatmaps.

    Combines focal loss weighting with KL divergence to handle
    the extreme class imbalance in heatmaps.

    Args:
        pred_heatmap (torch.Tensor): Predicted heatmap (N, C, H, W).
        target_heatmap (torch.Tensor): Target heatmap (N, C, H, W).
        weight (torch.Tensor, optional): Element-wise weights.
        alpha (float): Focal loss alpha parameter for positive samples.
        beta (float): Focal loss beta parameter for negative samples.
        temperature (float): Temperature scaling.
        reduction (str): Reduction method.
        avg_factor (int, optional): Average factor.

    Returns:
        torch.Tensor: Focal KL loss.
    """
    # Apply sigmoid
    # pred_prob = torch.softmax(pred_heatmap / temperature)
    # target_prob = torch.softmax(target_heatmap / temperature)

    pred_prob = pred_heatmap / temperature
    target_prob = target_heatmap / temperature

    # Clamp probabilities
    eps = 1e-6
    pred_prob = torch.clamp(pred_prob, min=eps, max=1 - eps)
    target_prob = torch.clamp(target_prob, min=eps, max=1 - eps)

    # Calculate KL divergence
    kl_pos = target_prob * torch.log(target_prob / pred_prob)
    kl_neg = (1 - target_prob) * torch.log((1 - target_prob) / (1 - pred_prob))
    kl_loss = kl_pos + kl_neg

    # Apply focal weighting
    # For positive samples (target ≈ 1)
    pos_weight = torch.pow(1 - pred_prob, alpha) * target_prob
    # For negative samples (target ≈ 0)
    neg_weight = torch.pow(pred_prob, beta) * (1 - target_prob)

    focal_weight = pos_weight + neg_weight
    focal_kl_loss = focal_weight * kl_loss

    # Apply additional weights
    if weight is not None:
        focal_kl_loss = focal_kl_loss * weight

    # Reduce loss
    return weight_reduce_loss(focal_kl_loss, None, reduction, avg_factor)


def find_local_maxima_robust(target_heatmap, eps=1e-5):
    # 使用最大池化找到局部最大值点
    local_max = F.max_pool2d(target_heatmap, kernel_size=3, stride=1, padding=1)
    is_local_max = (local_max == target_heatmap).float()

    # 峰值的理想值是1，我们给一个很小的容差
    is_peak_value = (target_heatmap > 1.0 - eps).float()

    # 结合两个条件
    pos_mask = is_local_max * is_peak_value
    return pos_mask


def gaussian_focal_kl_loss(pred_heatmap,
                           target_heatmap,
                           weight=None,
                           alpha=1.0,  # 预测概率的Focal Loss参数 (正样本)
                           beta=1.0,  # 预测概率的Focal Loss参数 (负样本)
                           gamma=4.0,  # 目标概率的Gaussian Focal Loss参数 (负样本)
                           temperature=1.0,
                           reduction='mean',
                           avg_factor=None,
                           eps=1e-6):
    """
    结合了Focal KL Loss和Gaussian Focal Loss思想的损失函数。

    Args:
        pred_heatmap (torch.Tensor): 预测热图 (N, C, H, W)。
        target_heatmap (torch.Tensor): 目标高斯热图 (N, C, H, W)。
        weight (torch.Tensor, optional): 像素级权重。
        alpha (float): 正样本的Focal Loss参数 (基于预测值)。
        beta (float): 负样本的Focal Loss参数 (基于预测值)。
        gamma (float): 负样本的Gaussian Loss参数 (基于目标值)。
        temperature (float): 温度缩放。
        reduction (str): 聚合方式。
        avg_factor (int, optional): 归一化因子。
        eps (float): 防止log(0)的小常数。

    Returns:
        torch.Tensor: 损失值。
    """
    # 为了简化，这里我们假设输入已经是概率值 (0-1)，或者需要先经过sigmoid
    # 在原始代码中，这一步是在KL散度计算之前的，我们保持一致
    # pred_prob = torch.sigmoid(pred_heatmap / temperature)
    pred_prob = pred_heatmap / temperature
    target_prob = target_heatmap  # 目标热图通常已经是0-1

    # 定义正样本为目标热图值为1的点 (这是CornerNet/CenterNet的典型做法)
    # pos_mask = (target_prob == 1).float()
    # pos_mask = torch.eq(target_prob, target_prob.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).float()
    pos_mask = find_local_maxima_robust(target_prob, eps=eps)

    # Clamp probabilities to avoid log(0) or log(1) issues
    pred_prob = torch.clamp(pred_prob, min=eps, max=1 - eps)
    target_prob = torch.clamp(target_prob, min=eps, max=1 - eps)

    # 1. 计算基础的KL散度损失
    kl_pos = target_prob * torch.log(target_prob / pred_prob)
    kl_neg = (1 - target_prob) * torch.log((1 - target_prob) / (1 - pred_prob))
    kl_loss = kl_pos + kl_neg

    # 2. 计算新的、结合了两种思想的Focal权重
    # 正样本权重 (与之前相同)
    pos_focal_weight = torch.pow(1 - pred_prob, alpha)

    # 负样本权重 (核心修改之处)
    neg_gaussian_weight = torch.pow(1 - target_prob, gamma)  # 来自Gaussian Focal Loss的思想
    neg_focal_weight = torch.pow(pred_prob, beta)  # 来自Focal Loss的思想

    # 将权重分别应用到KL散度的正负项上，逻辑更清晰
    # weighted_pos_loss = pos_focal_weight * kl_pos
    # weighted_neg_loss = neg_gaussian_weight * neg_focal_weight * kl_neg
    # combined_loss = (weighted_pos_loss + weighted_neg_loss)


    # 根据pos_mask，将两种权重应用到对应的像素点上
    # pos_mask为1的地方，使用pos_weight；为0的地方，使用neg_weight
    # pixel_wise_weight = pos_mask * pos_focal_weight + (1 - pos_mask) * neg_gaussian_weight * neg_focal_weight
    #
    # final_loss = pixel_wise_weight * kl_loss


    pos_weight = pos_mask * pos_focal_weight
    neg_weight = (1 - pos_mask) * neg_gaussian_weight * neg_focal_weight
    final_loss = pos_weight * kl_pos + neg_weight * kl_neg


    # 3. 应用外部权重并进行聚合
    if weight is not None:
        # combined_loss = combined_loss * weight
        final_loss = final_loss * weight

    return weight_reduce_loss(final_loss, None, reduction, avg_factor)


@MODELS.register_module()
class WeightKLLoss(nn.Module):
    """KL Divergence Loss for CenterNet Heatmaps.

    This loss is designed specifically for CenterNet-style heatmap regression,
    where the target is a Gaussian heatmap and we want to match the predicted
    distribution to the target distribution.

    Args:
        use_focal (bool): Whether to use focal weighting. Defaults to True.
        alpha (float): Focal loss alpha for positive samples. Defaults to 2.0.
        beta (float): Focal loss beta for negative samples. Defaults to 4.0.
        temperature (float): Temperature for distribution softening. Defaults to 1.0.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_focal=True,
                 alpha=1.0,
                 beta=1.0,
                 gamma=4.0,
                 temperature=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(WeightKLLoss, self).__init__()
        self.use_focal = use_focal
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted heatmap with shape (N, C, H, W).
            target (torch.Tensor): Target heatmap with shape (N, C, H, W).
            weight (torch.Tensor, optional): Element-wise weights.
            avg_factor (int, optional): Average factor for normalization.
            reduction_override (str, optional): Override reduction method.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        reduction = reduction_override if reduction_override else self.reduction

        if self.use_focal:
            # loss = focal_kl_loss(
            #     pred, target, weight,
            #     alpha=self.alpha,
            #     beta=self.beta,
            #     temperature=self.temperature,
            #     reduction=reduction,
            #     avg_factor=avg_factor
            # )
            loss = gaussian_focal_kl_loss(
                pred, target, weight,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                temperature=self.temperature,
                reduction=reduction,
                avg_factor=avg_factor
            )
        else:
            loss = centernet_heatmap_kl_loss(
                pred, target, weight,
                temperature=self.temperature,
                reduction=reduction,
                avg_factor=avg_factor
            )

        return self.loss_weight * loss


@MODELS.register_module()
class SoftTargetCrossEntropyLoss(nn.Module):
    """Cross Entropy Loss for soft targets, such as Gaussian heatmaps.

    This loss function is the correct and stable way to implement a KL-like
    objective for heatmaps without suffering from negative loss values. It is
    mathematically equivalent to minimizing KL divergence for optimization purposes.

    Args:
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,  # weight is applied element-wise on the loss map
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted heatmap logits, shape (N, C, H, W).
            target (torch.Tensor): Target heatmap, shape (N, C, H, W).
                                   Does NOT need to be a normalized distribution.
            weight (torch.Tensor, optional): Element-wise weight, shape (N, C, H, W).
            avg_factor (int, optional): Average factor for normalization.
            reduction_override (str, optional): Override reduction method.

        Returns:
            torch.Tensor: Computed loss.
        """
        assert pred.shape == target.shape
        reduction = reduction_override if reduction_override else self.reduction

        # 针对该问题的正确损失函数：带Logits的二元交叉熵。
        # 它在内部对 'pred' 应用 sigmoid 函数，然后计算BCE。
        loss_map = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        # 3. 应用逐元素的权重（如果提供的话）
        #    这可以用来实现 Focal Loss 的思想，例如给正样本更高的权重。
        if weight is not None:
            loss_map = loss_map * weight

        # 4. 根据 reduction 参数聚合损失
        #    我们先对C,H,W维度求和，得到每个样本的loss，然后再聚合
        #    或者直接使用 weight_reduce_loss，它能更灵活地处理
        loss = weight_reduce_loss(loss_map, weight=None, reduction=reduction, avg_factor=avg_factor)

        return self.loss_weight * loss


@MODELS.register_module()
class CenterNetDistillationLoss(nn.Module):
    """Knowledge Distillation Loss for CenterNet.

    Combines KL divergence between teacher and student heatmaps
    with the original CenterNet focal loss on ground truth.

    Args:
        alpha (float): Weight for KL loss (teacher knowledge). Defaults to 0.7.
        beta (float): Weight for focal loss (ground truth). Defaults to 0.3.
        temperature (float): Temperature for knowledge distillation. Defaults to 3.0.
        kl_focal_alpha (float): Focal alpha for KL loss. Defaults to 2.0.
        kl_focal_beta (float): Focal beta for KL loss. Defaults to 4.0.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_weight (float): Overall loss weight. Defaults to 1.0.
    """

    def __init__(self,
                 alpha=0.7,
                 beta=0.3,
                 temperature=3.0,
                 kl_focal_alpha=2.0,
                 kl_focal_beta=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(CenterNetDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

        # KL loss for teacher-student knowledge transfer
        self.kl_loss = WeightKLLoss(
            use_focal=True,
            alpha=kl_focal_alpha,
            beta=kl_focal_beta,
            temperature=temperature,
            reduction=reduction,
            loss_weight=1.0
        )

        # Original CenterNet focal loss for ground truth
        # You might need to import this from mmdet or implement it
        try:
            from mmdet.models.losses import FocalLoss
            self.focal_loss = FocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction=reduction,
                loss_weight=1.0
            )
        except ImportError:
            # Fallback implementation
            self.focal_loss = self._simple_focal_loss

    def _simple_focal_loss(self, pred, target, weight=None, avg_factor=None):
        """Simple focal loss implementation for fallback."""
        pred_sigmoid = torch.sigmoid(pred)
        target_float = target.float()

        # Focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(pred, target_float, reduction='none')
        p_t = pred_sigmoid * target_float + (1 - pred_sigmoid) * (1 - target_float)
        focal_weight = (1 - p_t) ** 2.0  # gamma = 2.0
        focal_loss = focal_weight * ce_loss

        return weight_reduce_loss(focal_loss, weight, self.reduction, avg_factor)

    def forward(self,
                student_pred,
                teacher_pred,
                target,
                weight=None,
                avg_factor=None,
                **kwargs):
        """Forward function.

        Args:
            student_pred (torch.Tensor): Student heatmap predictions (N, C, H, W).
            teacher_pred (torch.Tensor): Teacher heatmap predictions (N, C, H, W).
            target (torch.Tensor): Ground truth heatmaps (N, C, H, W).
            weight (torch.Tensor, optional): Loss weights.
            avg_factor (int, optional): Average factor.

        Returns:
            torch.Tensor: Combined distillation loss.
        """
        # KL loss between student and teacher
        kl_loss = self.kl_loss(
            student_pred,
            teacher_pred,
            weight=weight,
            avg_factor=avg_factor
        )

        # Focal loss on ground truth
        if hasattr(self.focal_loss, '__call__'):
            focal_loss = self.focal_loss(
                student_pred,
                target,
                weight=weight,
                avg_factor=avg_factor
            )
        else:
            focal_loss = self._simple_focal_loss(
                student_pred,
                target,
                weight=weight,
                avg_factor=avg_factor
            )

        # Combine losses
        total_loss = (self.alpha * kl_loss + self.beta * focal_loss) * self.loss_weight

        return total_loss



"""
# 配置文件使用示例：

# 1. 基本CenterNet KL损失
model = dict(
    type='CenterNet',
    # ... other configs ...
    bbox_head=dict(
        type='CenterNetHead',
        # ... other configs ...
        loss_center_heatmap=dict(
            type='CenterNetKLLoss',
            use_focal=True,
            alpha=2.0,
            beta=4.0,
            temperature=1.0,
            loss_weight=1.0
        )
    )
)

# 2. 知识蒸馏配置
model = dict(
    type='CenterNetDistiller',  # 假设你有一个蒸馏器
    # ... teacher and student configs ...
    loss_distill=dict(
        type='CenterNetDistillationLoss',
        alpha=0.7,  # Teacher knowledge weight
        beta=0.3,   # Ground truth weight
        temperature=3.0,
        loss_weight=1.0
    )
)

# 3. 训练配置示例
train_cfg = dict(
    # ... other training configs ...
    distillation=dict(
        teacher_config='path/to/teacher/config.py',
        teacher_checkpoint='path/to/teacher/weights.pth'
    )
)
"""