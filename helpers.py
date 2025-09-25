import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 常用辅助函数与组件：扩散调度、时间位置编码、加权损失等。


class SinusoidalPosEmb(nn.Module):
    # 正弦位置编码：将标量时间步映射到高维周期特征
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # 频率衰减因子
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 构造不同频率波段
        emb = x[:, None] * emb[None, :]  # 广播到 batch 维度
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 拼接正弦和余弦分量
        return emb


def extract_into_tensor(a, t, x_shape):
    # 从一维调度表 a 中依据时间索引 t 提取值并 reshape 成与 x 同形状的前缀
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    Cosine schedule in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 根据累计乘积反推 β 序列
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)  # 限制上界避免数值不稳定
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    # 线性调度：β 从较小值平滑增加到较大值
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    # Variance Preserving 调度：控制 β 使得扩散过程保持方差稳定
    t = np.arange(1, timesteps + 1)
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / timesteps - 0.5 * (b_max - b_min) * (2 * t - 1) / timesteps ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class WeightedLoss(nn.Module):
    # 带权重的损失抽象基类，子类只需实现 _loss
    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss):

    def _loss(self, predict_values, target_values):
        # L1 损失：绝对误差
        return torch.abs(predict_values - target_values)


class WeightedL2(WeightedLoss):

    def _loss(self, predict_values, target_values):
        # L2 损失：逐元素 MSE（不做平均，交给父类加权）
        return F.mse_loss(predict_values, target_values, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2
}

class EMA:
    # 指数滑动平均，用于参数平滑（可复制 EMA 模型）
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new  # EMA 公式：旧权重衰减 + 新值增量
