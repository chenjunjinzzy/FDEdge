import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract_into_tensor,
    Losses
)


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, beta_schedule='linear', denoising_steps=5,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度
        self.model = model  # 预测噪声/动作的神经网络模型

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(denoising_steps)  # 线性β调度（扩散强度随步数线性变化）
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(denoising_steps)  # 余弦β调度（更平滑的扩散过程）
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(denoising_steps)  # Variance Preserving 调度

        alphas = 1. - betas  # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # \bar{α}_t = ∏_{i=1..t} α_i（累计乘积）
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # \bar{α}_{t-1}，t=0时为1

        self.n_timesteps = int(denoising_steps)  # 去噪步数（扩散链长度）
        self.clip_denoised = clip_denoised  # 是否对重建的x0进行裁剪
        self.predict_epsilon = predict_epsilon  # 模型输出是否为噪声ε（True）或直接为x0（False）

        self.register_buffer('betas', betas)  # 缓存β序列到buffer（随模型一起移动设备/保存）
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # 缓存\bar{α}_t
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # 缓存\bar{α}_{t-1}

        # 扩散分布 q(x_t | x_{t-1}) 及相关项的预计算
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # √\bar{α}_t
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  # √(1-\bar{α}_t)
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))  # log(1-\bar{α}_t)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))  # √(1/\bar{α}_t)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # √(1/\bar{α}_t - 1)

        # 后验分布 q(x_{t-1} | x_t, x_0) 的方差
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 论文中标准公式
        self.register_buffer('posterior_variance', posterior_variance)  # 缓存后验方差

        # 因t=0时方差为0，这里对log方差做下限裁剪以避免数值问题
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))  # log方差并做裁剪
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # 后验均值系数1
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))  # 后验均值系数2

        self.loss_fn = Losses[loss_type]()  # 损失函数（如L1/L2），由配置选择

    # def predict_start_from_noise(self, x_t, t, noise):
    #     """
    #         if self.predict_epsilon, model output is (scaled) noise;
    #         otherwise, model predicts x0 directly
    #     """
    #     if self.predict_epsilon:
    #         return (
    #                 extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
    #                 extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    #         )
    #     else:
    #         return noise
    # 上面是原始可切换实现：predict_epsilon=True 时根据ε重建x0；False时网络直接输出x0

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )  # 依据x_t与预测噪声ε重建x0（常用设置）

    def predict_start_from_t_v(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                    extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )  # 与上式一致：根据ε推回x0
        else:
            return noise  # 若模型直接预测x0，这里返回“noise”变量（此处语义即x0）

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # 后验均值：线性组合x0与x_t
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)  # 按t取对应后验方差
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)  # 取裁剪后的log方差
        return posterior_mean, posterior_variance, posterior_log_variance_clipped  # 返回后验参数三元组

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))  # 用模型预测噪声并重建x0
        if self.clip_denoised:
            x_recon.clamp_(-1, 1)  # 可选：限制重建x0的数值范围
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)  # 计算p的均值和方差（由q的后验得到）
        return model_mean, posterior_variance, posterior_log_variance  # 作为采样分布参数

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device  # 批大小与设备
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)  # 得到条件分布参数
        probs = torch.randn_like(x)  # 采样用标准高斯噪声
        # probs = x  # （保留原作者调试痕迹）
        # 当 t == 0 时不再加噪声（最后一步输出确定性）
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # t=0屏蔽噪声项
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * probs  # 采样：均值 + σ * 噪声

    # def p_sample_loop(self, state, shape):
    #     device = self.betas.device
    #     batch_size = shape[0]
    #     x = torch.randn(shape, device=device)
    #     for i in reversed(range(0, self.n_timesteps)):
    #         timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
    #         x = self.p_sample(x, timesteps, state)
    #     return x
    # 上面是根据给定shape从纯噪声开始逐步去噪的版本（保留以便参考）

    def p_sample_loop(self, state, latent_action_probs):
        device = self.betas.device  # 当前参数所在设备
        batch_size = state.shape[0]  # 批大小
        # x = torch.randn(batch_size, self.action_dim, device=device)  # 也可从纯噪声初始化
        # x = latent_action_probs  # 或直接用潜在先验初始化
        x = torch.randn_like(latent_action_probs)  # 这里选择与先验同形状的高斯噪声初始化
        for i in reversed(range(0, self.n_timesteps)):  # 从t=T-1递减到0逐步去噪
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)  # 当前步t张量
            x = self.p_sample(x, timesteps, state)  # 单步采样更新x
        return x  # 返回最终动作的未归一化logits（或值）

    # def sample(self, state, *args, **kwargs):
    #     batch_size = state.shape[0]
    #     shape = (batch_size, self.action_dim)
    #     action = self.p_sample_loop(state, shape, *args, **kwargs)
    #     return F.softmax(action, dim=-1)
    # 另一种采样接口：根据shape从噪声开始生成动作分布

    def sample(self, state, latent_action_probs, *args, **kwargs):
        action = self.p_sample_loop(state, latent_action_probs, *args, **kwargs)  # 通过扩散链生成动作logits
        return F.softmax(action, dim=-1)  # 对动作维做softmax，得到概率分布

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)  # 若未提供噪声则采样标准高斯

        sample = (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )  # 前向扩散：根据x0与噪声生成x_t
        return sample  # 返回x_t

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)  # 训练时加入的噪声ε
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 合成带噪样本x_t
        x_recon = self.model(x_noisy, t, state)  # 模型预测ε或x0
        assert noise.shape == x_recon.shape  # 形状一致性检查
        if self.predict_epsilon:
            return self.loss_fn(x_recon, noise, weights)  # 若预测ε，则回归目标为真实噪声
        else:
            return self.loss_fn(x_recon, x_start, weights)  # 若预测x0，则回归目标为原始x0

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)  # 批大小
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()  # 随机采样扩散步t
        return self.p_losses(x, state, t, weights)  # 计算训练损失

    def forward(self, state, latent_action_probs, *args, **kwargs):
        return self.sample(state, latent_action_probs, *args, **kwargs)  # 前向：输出动作概率分布
