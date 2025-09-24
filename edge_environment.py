import numpy as np
import collections

import torch
from scipy import stats


class OffloadEnvironment:
    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # 输入数据
        self.n_tasks = num_tasks  # 移动设备/任务数量
        self.n_BSs = num_BSs  # 基站/边缘服务器数量
        self.time_slots = time_slots_  # 时间槽总数
        self.state_dim = 2 + self.n_BSs  # 系统状态维度=2（任务比特、该任务计算需求）+各ES队列长度
        self.action_dim = num_BSs
        self.duration = 1  # 每个时间槽长度（秒）
        self.ES_capacities = es_capacities  # 各ES计算能力（十亿周期/秒）
        np.random.seed(5)
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])  # 到各基站的上行传输速率范围[400,500] Mbit/s
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])  # 每Mbit所需计算量（Gcycles/Mbit）

        # 存储系统中各时间槽到达的任务比特数
        self.tasks_bit = []
        self.min_bit = bit_range[0]  # 任务比特下限
        self.max_bit = bit_range[1]  # 任务比特上限

        # 各ES在各时间槽的处理队列工作量（Gcycles）
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])
        # 当前槽内、处理下一个任务前的累计队列工作量（Gcycles）
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])
        self.wait_delay = 0  # 每个任务的初始等待时延

        # 预生成的潜在动作概率空间（可作为先验/探索用）
        self.latent_action_prob_space = np.random.normal(size=[self.time_slots, self.n_tasks, self.action_dim])

    def reset_env(self, tasks_bit):
        self.tasks_bit = tasks_bit  # 初始化环境中的全部任务到达比特
        # 重置各ES处理队列工作量（Gcycles）
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])
        # 重置当前槽内累计的“处理前”队列工作量
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])
        self.wait_delay = 0  # 重置任务等待时延

    # 执行任务卸载以得到：
    # (1) 服务时延；
    # (2) 各ES队列工作量的更新；
    def step(self, t, n, action):
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]
        # 计算第n个任务的传输+计算时延
        tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                            self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])

        # 总服务时延=等待+传输+计算
        delay = tran_comp_delays + self.wait_delay
        reward = - delay  # 奖励取负时延（时延越小奖励越大）
        # 在当前槽内、为处理下一个任务前，累加所选ES的队列工作量
        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][n]

        # 观测下一状态与潜在动作（供策略决策）
        if n == len(self.tasks_bit[t]) - 1:
            next_state = np.hstack([self.tasks_bit[t + 1][0],
                                    self.comp_density[0] * self.tasks_bit[t+1][0],
                                    self.proc_queue_len[t + 1]])
            next_potential_action = self.latent_action_prob_space[t + 1][0]
        else:
            next_state = np.hstack([self.tasks_bit[t][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][n+1],
                                    self.proc_queue_len[t]])
            next_potential_action = self.latent_action_prob_space[t][n + 1]

        return next_state, next_potential_action, reward, delay

    # 在下一个时间槽开始时更新各ES队列长度（扣除本槽可处理的计算量）
    def update_proc_queues(self, t):
        for b_ in range(self.n_BSs):
            self.proc_queue_len[t + 1][b_] = np.max(
                [self.proc_queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.ES_capacities[b_] * self.duration, 0])
