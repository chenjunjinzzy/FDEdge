from edge_environment import OffloadEnvironment
from fdsac_model import *
import matplotlib.pyplot as plt


def FDScheduling_algorithm():
    # 环境参数初始化
    NUM_BSs = 10  # 基站/边缘服务器数量
    NUM_TASKS_max = 100  # 主节点可接收的最大任务数（用于采样上限）
    BIT_RANGE = [10, 40]  # 任务大小范围（单位：Mbit）
    NUM_TIME_SLOTS = 100  # 时间槽数量（一次训练轮中包含的离散时刻数）
    ES_capacity_max = 50  # 边缘服务器计算能力上限（Gcycles/s）
    np.random.seed(2)
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])  # 随机生成各边缘服务器的计算能力（Gcycles/s）

    # 深度强化学习模型参数
    actor_lr = 1e-4  # Actor（扩散决策网络）学习率
    critic_lr = 1e-3  # Critic 学习率
    alpha = 0.05  # 熵正则温度系数（控制探索强度）
    alpha_lr = 3e-4  # 熵参数学习率
    episodes = 5  # 训练的总轮数
    denoising_steps = 5  # 扩散调度模型的去噪步数
    hidden_dim = 128  # 神经网络隐藏层宽度
    gamma = 0.95  # 折扣因子
    tau = 0.005  # 软更新系数（目标网络参数平滑更新）
    train_buffer_size = 10000  # 经验回放池容量
    batch_size = 64  # 每次训练的采样批量
    target_entropy = -1  # 目标熵
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 训练设备选择（优先GPU）

    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)  # 生成环境
    agent = FDSAC(env.state_dim, hidden_dim, env.action_dim, actor_lr, critic_lr, alpha, alpha_lr,
                  target_entropy, tau, gamma, denoising_steps, device)  # 初始化智能体
    train_buffer = ReplayBuffer(train_buffer_size)  # 初始化经验回放池

    # =============== 基于 FDSAC 的在线任务调度与并行训练 ===================
    average_delays = []  # 各训练轮的平均服务时延
    for i_episode in range(episodes):
        # ======= 生成环境中的到达任务 ===========
        arrival_tasks = []  # 保存每个时间槽的任务到达集合
        for i in range(env.time_slots):
            task_dim = np.random.randint(1, env.n_tasks + 1)  # 当前时间槽内任务数量
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))  # 任务大小采样

        env.reset_env(arrival_tasks)  # 重置环境（装载本轮的任务到达序列）
        episode_delays = []  # 存储本轮所有任务的服务时延
        exe_count = 0  # 任务调度执行计数
        for t in range(env.time_slots - 1):
            # ================ 在线任务调度 ====================
            task_set_len = len(env.tasks_bit[t])  # 当前时间槽的任务数
            for n in range(task_set_len):
                state = np.hstack([env.tasks_bit[t][n],
                                   env.comp_density[n] * env.tasks_bit[t][n],
                                   env.proc_queue_len[t]])  # 状态：任务比特、该任务计算需求、各ES当前队列
                latent_action_probs = env.latent_action_prob_space[t][n]  # 潜在动作概率（先验/探索参考）
                action, action_probs = agent.take_action(state, latent_action_probs)  # Actor 生成卸载决策与更新后的概率
                env.latent_action_prob_space[t][n] = action_probs  # 回写潜在动作空间（供后续使用）
                next_state, next_latent_action, reward, delay = env.step(t, n, action)  # 执行卸载，得到下一状态与时延/奖励
                train_buffer.add(state, action, latent_action_probs, reward, next_state, next_latent_action)  # 存入经验
                episode_delays.append(delay)  # 记录单任务服务时延
                exe_count = exe_count + 1  # 更新执行计数
            env.update_proc_queues(t)  # 时间槽结束：更新所有边缘服务器的处理队列

            # ============= 并行网络训练 ===============
            # 经验池样本数大于 500 即可开始训练；也可按步数间隔触发（见注释的原始写法）
            # if train_buffer.size() > 500 and exe_count % batch_size == 0:  # 按步数间隔训练
            if train_buffer.size() > 500:  # 达到最小样本阈值后进行训练
                b_s, b_a, b_p, b_r, b_ns, b_np = train_buffer.sample(batch_size)  # 批量采样
                transition_dict = {'states': b_s, 'actions': b_a, 'latent_action_probs': b_p,
                                   'rewards': b_r, 'next_states': b_ns, 'next_latent_action_probs': b_np}  # 整理批数据
                agent.update(transition_dict)  # 更新网络参数

        average_delays.append(np.mean(episode_delays))  # 记录本轮的平均服务时延
        print({'Episode': '%d' % (i_episode + 1), 'average delay': '%.4f' % average_delays[-1]})  # 输出训练进度

    print('============ 完成 FDEdge 方法的任务卸载与模型训练 ==========')  # 训练完成提示

    # 保存并绘制：平均时延随训练轮数变化
    episodes_list = list(range(len(average_delays)))  # 横轴：轮数
    np.savetxt('results/AveDelay_FDEdge_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_steps' + str(denoising_steps) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')  # 保存CSV结果
    plt.figure(1)
    plt.plot(episodes_list, average_delays)  # 曲线：平均时延
    plt.ylabel('Average service delay')  # 纵轴标签（可按需改成中文：平均服务时延）
    plt.xlabel('Episode')  # 横轴标签（可按需改成中文：训练轮数）
    plt.savefig('results/AveDelay_FDEdge_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_steps' + str(denoising_steps) +
                '_episode' + str(episodes) + '.png')  # 保存PNG图
    plt.close()


if __name__ == '__main__':
    FDScheduling_algorithm()  # 入口：运行调度与训练
