from feedback_diffusion import *  # 扩散策略核心实现
from helpers import SinusoidalPosEmb  # 正弦时间位置编码
import numpy as np
import torch
import torch.nn as nn
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        # 使用有界 deque 按先进先出方式缓存轨迹片段
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, latent_action_probs, reward, next_state, next_latent_action_probs):
        # 将一次交互的全部信息打包入缓冲区
        self.buffer.append((state, action, latent_action_probs, reward, next_state, next_latent_action_probs))

    def sample(self, batch_size):
        # 随机采样批量经验，用于离线训练更新
        transitions = random.sample(self.buffer, batch_size)
        state, action, latent_action_probs, reward, next_state, next_latent_action_probs = zip(*transitions)
        return np.array(state), action, np.array(latent_action_probs), reward, np.array(next_state), np.array(
            next_latent_action_probs)
        # return np.array(state), action, latent_action_probs, reward, np.array(next_state), next_latent_action_probs

    def size(self):
        # 返回当前缓存的样本数量
        return len(self.buffer)

    def clear(self):
        # 清除所有缓存经验
        self.buffer.clear()


class MLP_PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, t_dim=16):
        super(MLP_PolicyNet, self).__init__()
        # 将扩散步骤转为嵌入，提供时间条件信号
        self.time_FCN = SinusoidalPosEmb(t_dim)
        #
        # self.time_FCN1 = SinusoidalPosEmb(t_dim)
        # self.time_FCN2 = nn.Linear(t_dim, t_dim * 2)
        # self.time_FCN3 = nn.Linear(t_dim * 2, t_dim)
        # self.time_FCN = nn.Sequential(
        #     SinusoidalPosEmb(t_dim),
        #     nn.Linear(t_dim, t_dim * 2),
        #     nn.ReLU,
        #     nn.Linear(t_dim * 2, t_dim)
        # )

        # MLP 接收噪声、时间嵌入和环境状态，输出动作 logits
        self.fc1 = nn.Linear(state_dim + action_dim + t_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time_step, state):
        # 时间步嵌入与状态、当前噪声拼接后送入网络
        t = self.time_FCN(time_step)
        # t = self.time_FCN1(time_step)
        # t = F.relu(self.time_FCN2(t))
        # t = self.time_FCN3(t)

        state = state.reshape(state.size(0), -1)
        x = torch.cat([x, t, state], dim=1)
        # x = self.mid_layer(x)
        # return self.final_layer(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class QValueNet(torch.nn.Module):
    """ Two hidden layers """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # 估计每个离散动作的 Q 值，用于双重 Q 学习
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FDSAC:
    ''' 处理离散动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha,
                 alpha_lr, target_entropy, tau, gamma, denoising_steps_, device):
        # 扩散 Actor：条件扩散链生成动作分布
        self.actor = Diffusion(state_dim=state_dim,
                               action_dim=action_dim,
                               model=MLP_PolicyNet(state_dim, hidden_dim, action_dim),
                               beta_schedule='vp',
                               denoising_steps=denoising_steps_).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # 第一个 Q 网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # 第二个 Q 网络

        self.target_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标网络
        self.target_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标网络

        # 初始化目标网络参数，保持与主 Q 网络一致
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_2.load_state_dict(self.critic_2.state_dict())

        # 三个主要网络的优化器配置
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.history_loss = []

    def take_action(self, state, latent_actions):
        # 将单步状态与潜在动作分布转换为批量张量
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        latent_actions = torch.tensor(np.array([latent_actions]), dtype=torch.float).to(self.device)
        probs = self.actor(state, latent_actions)
        action_list = probs.tolist()
        action = np.argmax(action_list[0])  # 贪心挑选概率最大的动作索引
        return action, probs.detach().cpu().numpy()  # Get the action index and latent_ation_probs
        # action_dist = torch.distributions.Categorical(probs)  # Normalization
        # action = action_dist.sample()  # Get the index tensor of the maximal probability
        # return action.item(), probs.detach().cpu().numpy()  # Get the action index and latent_ation_probs

    # Calulcate the target Q values
    # Using the actor output, V output, and target V output with the input of reward and next state.
    def calc_target_q(self, rewards, next_states, next_latent_actions):
        next_probs = self.actor(next_states, next_latent_actions)  # 下一状态下的策略分布
        next_log_probs = torch.log(next_probs + 1e-8)  # 1e-8 防止 log(0) 导致 NaN

        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)  # 策略的熵项

        target1_q1_value = self.target_1(next_states)
        target2_q2_value = self.target_2(next_states)

        min_q_value = torch.sum(next_probs * torch.min(target1_q1_value, target2_q2_value), dim=1, keepdim=True)
        next_value = min_q_value + self.log_alpha.exp() * entropy  # 双 Q 最小化 + 熵调节
        # q_target = rewards + self.gamma * next_value * (1 - dones)
        q_target = rewards + self.gamma * next_value
        return q_target

    def soft_update(self, net, target_net):
        # 执行 Polyak 平滑更新，缓解训练震荡
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # 将经验批量转换为张量并搬到相应设备
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)  # 动作不再是float类型
        latent_actions = torch.tensor(transition_dict['latent_action_probs'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        next_latent_actions = torch.tensor(transition_dict['next_latent_action_probs'], dtype=torch.float).to(
            self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # --- 更新两个 Q 网络 ---
        target_q_values = self.calc_target_q(rewards, next_states, next_latent_actions).detach()
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, target_q_values.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, target_q_values.detach()))
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # --- 更新 Actor 网络 ---
        probs = self.actor(states, latent_actions)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 自适应调整温度系数 α ---
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # print(alpha_loss.item())
        # print(entropy[-1])
        # print(self.log_alpha.exp())

        # 软更新目标网络并记录训练损失
        self.soft_update(self.critic_1, self.target_1)
        self.soft_update(self.critic_2, self.target_2)

        self.history_loss.append(critic_2_loss.item())
