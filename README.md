# Edge-enabled AI-Generated Content using Latent Action Diffusion-based Task Scheduling
This code is a implementation of our paper "Edge-enabled AI-Generated Content using Latent Action Diffusion-based Task Scheduling", submitted in INFOCOM 2025.
This paper presents a novel method named LAD-TS that uses a latent action diffusion strategy to optimize task sheduling policy.

To run this code, please install packages: Pytorch 2.1.0., NumPy, and matplotlib.

The code of our method mainly consists of three files: diffusion.py, SAC_diffusion.py, environment.py, and main.py.

The main.py file is the main code. User should run this code to acheive the experimental results.

The environment.py inculdes the code for MEC environment. In this file, some environment parameters such as ESs' computing capacities, task size, and transmission rate can be set by user.

The diffusion.py includes the code for reverse diffusion model. 

The SAC_diffusion.py includes the code for the LADN model based on the Soft-Actor-Critic (SAC) framework. 

# Results
Some experimental results of each method are stored in the corresponding results directory. User can run the corresponding main.py file to achieve these results again. Note that sometimes, the results may be some devivations. However, user can achieve the better results by running more times.

We implement two well-known baselines (i.e., DQN-TS [1] and SAC-TS [2]) and a heuristic optimal (i.e., Opt-TS) baseline in our experiments. The DQN-TS and SAN-TS baselines are implemented based on the code of DQN and SAC methods which can grab a public copy. We implement the Opt-TS baseline by enumerating all action spaces to select the most suitable ESs to collaboratively process each task. Note that for fairness, all the methods are satisfied with the same constraints and parameters in the experiment.

[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,”
nature, vol. 518, no. 7540, pp. 529–533, 2015.

[2] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proceedings of the 35th International Conference on Machine Learning (PMLR), vol. 80. PMLR, 2018, pp. 1861–1870.

# Paramters setting
All the parameters in current codes are set by default. Hence, user can ajust the parameters such as the number of traning episode, the number of tasks, the required quality of AIGC, the ESs' capacities, and the number of BSs. 
