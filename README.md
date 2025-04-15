# FDEdge Implementation
This repo is an implementation of our paper "**Enhancing QoS in Collaborative Edge Systems with Feedback Diffusion Generative Scheduling**", **submitted to IEEE Transactions on Mobile Computing**. In this repo, we implement the proposed FDEdge method and baselines in our paper.

## I. FDEdge Framework
we propose the FDEdge method by designing a feedback diffusion model-based DRL framework, which can generate better task-scheduling solutions with multi-step decision-making.
<div align=center>
<img src="results/FDEdge_framework.jpg" width="500px">
</div>
The overall architecture of our FDEdge method, in which a Feedback Diffusion Network (FDN) model is designed and integrated into the actor. For each task arriving at the master node, the scheduler will offload it to a suitable ES for processing by the actor. The history data of task offloading is stored in an experienced pool. The FDN model is trained using the experience pool's history samples. The historical action probability $\boldsymbol{x}_{t,n}$ is recorded or updated in memory for feedback as the input of the diffusion process.

## II. Actor Structure
The actor structure is a multi-step decision-making process that is designed by an FDN model, a softmax unit, and a sampling unit.
<div align=center>
<img src="results/actor_structure.jpg" width="600px">
</div>
The actor structure with proposed FDN model. The actor input are the timestep $I$, potential action probability $\boldsymbol{x}_{t,n,I}$, and system state $\boldsymbol{s}_{t,n}$. The output is the action decision $\boldsymbol{a}_{t,n}$. The historical action probability $\boldsymbol{x}_{t,n,0}$ is stored (or updated) into the array $X_{t}[n]$.

## III. Comparison Performance 
<div align=center>
<img src="results/comparison_performance.jpg" width="350px">
</div>
Our FDEdge method achieves the lowest delay, outperforming 87.57%, 84.22%, 79.93%, and 63.28% compared to the Rand, SAC, DQN, and D2SAC methods, respectively, and closely approximates the optimal method's delay.

## IV. FDEdge's Code implementation
The code of the FDEdge method mainly includes the following four files: 

- `feedback_diffusion.py`: This file implements the feedback diffusion processing.

- `edge_environment.py`: This file implements the collaborative edge computing environment that is initialized according to the given parameters in fdedge_main.py. 

- `fdsac_model.py`: This file implements the Mulit-Layer Perception (MLP) network, Actor, Target network, Critic network, and Network training processing.

- `fdedge_main.py`: This file implements the FDEdge algorithm's procedure. In this file, some key environment and model parameters are given. User can set by yourself. For instance, you can set the variable NUM_TASKS = 100 that represents the number of tasks arrived to the master node is in the range [1, 100] at a time slot t.

### Test Usage
User can run the `fdedge_main.py` to achieve the corresponding experimental results.
```sh
python3 fdedge_main.py
```

### Install
To run this code, please install some key packages: torch, NumPy, and matplotlib

## V. Baselines Implementation
In our paper, we use four baselines: Rand, DQN, SAC, D2SAC, and Opt.
The baselines are implemented in the Baselines directory.

(1) Rand baseline: Rand is a traditional heuristic method that randomly selects an ES to process for each task offloading. This method is a classic offloading solution often used as a comparison in edge computing studies.

(2) DQN baseline: The Deep Q-Network (DQN) [1] is a widely recognized DRL method that has been successfully applied in various domains. In our experiments, we faithfully implement the DQN method for task scheduling as a baseline, ensuring that it uses the same setup and configuration as our proposed method. User can run the `dqn_main.py` to achieve experimental results. 

(3) SAC baseline: The SAC [2] a state-of-the-art DRL algorithm known for its stability and efficiency in continuous action spaces. We implement SAC for task scheduling as another baseline, maintaining consistency in the experimental setup for a fair comparison. User can run the `sac_main.py` to achieve experimental results.

(4) D2SAC [3] baseline: D2SAC is the state-of-the-art scheduling method based on the diffusion model. This code implementation can refer the [release code](https://github.com/Lizonghang/AGOD).

(5) Opt baseline: The Opt method selects the best ES for each task by exhaustively evaluating all possible action combinations, representing an upper bound on task scheduling performance in our experiments. However, while it provides the theoretical best solution, it is impractical in real-world scenarios, as it requires prior knowledge of the available computing and network resources for each ES. User can run the `opt_main.py` to achieve experimental results.

### Install
To run this code, please install some key packages: torch, NumPy, and matplotlib

# References
[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” nature, vol. 518, no. 7540, pp. 529–533, 2015. [Code](https://github.com/LiSir-HIT/Reinforcement-Learning/tree/main/Model/1.%20DQN)

[2] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proceedings of the 35th International Conference on Machine Learning (PMLR), vol. 80. PMLR, 2018, pp. 1861–1870. [Code](https://github.com/LiSir-HIT/Reinforcement-Learning/tree/main/Model/8.%20SAC_Discrete)

[3] H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao, “Diffusion-based reinforcement learning for edge-enabled ai-generated content services,” IEEE Transactions on Mobile Computing, 2024. [Code](https://github.com/Lizonghang/AGOD)

#### Citation
If our code can be used in your paper, please help cite:

@article{xu2024accelerating,
  title={Accelerating AIGC Services with Latent Action Diffusion Scheduling in Edge Networks},
  author={Xu, Changfu and Guo, Jianxiong and Lin, Wanyu and Zou, Haodong and Fan, Wentao and Wang, Tian and Chu, Xiaowen and Cao, Jiannong},
  journal={arXiv preprint arXiv:2412.18212},
  year={2024}
}

@inproceedings{xu2024phd,
  title={PhD Forum Abstract: Diffusion-based Task Scheduling for Efficient AI-Generated Content in Edge Networks},
  author={Xu, Changfu},
  booktitle={2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)},
  pages={333--334},
  year={2024},
  organization={IEEE}
}
