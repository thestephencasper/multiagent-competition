import sys
import os
import time
import multiprocessing
from multiprocessing import freeze_support
import numpy as np
import torch
from stable_baselines3.ppo import PPO
# from stable_baselines import PPO2
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_compete.utils import parse_args, POLICY_KWARGS_MLP, POLICY_KWARGS_LSTM
from gym_compete.envs import make_env, get_env_and_policy, simple_eval


if __name__ == '__main__':

    freeze_support()
    multiprocessing.set_start_method('spawn')
    sd = int(str(time.time()).replace('.', '')[-5:])
    torch.manual_seed(sd)
    args = parse_args()
    tac = args.agent1_ckpt if args.ta_i else args.agent0_ckpt  # training agent checkpoint
    fac = args.agent0_ckpt if args.ta_i else args.agent1_ckpt  # fixed agent checkpoint
    device = args.device
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ta_i)

    env_type, policy_alg, policy_kwargs, net_type = get_env_and_policy(args.env, args.use_sb3)
    env = SubprocVecEnv([make_env(env_type, fac, policy_alg, net_type, policy_kwargs, args, i) for i in range(args.n_envs)])
    eval_env = SubprocVecEnv([make_env(env_type, fac, policy_alg, net_type, policy_kwargs, args, 42)])

    if args.use_sb3:
        policy = policy_alg(net_type, env, ent_coef=args.ent_coef, policy_kwargs=policy_kwargs, learning_rate=args.lr,
                            batch_size=args.bs, n_epochs=args.n_epochs, n_steps=args.n_steps, device=device, seed=sd)
        if tac:
            policy.set_parameters(load_path_or_dict=(args.model_dir + tac), device=device)
    else:
        raise NotImplementedError
        # if tac:
        #     policy = policy_alg.load(args.model_dir + tac)
        # else:
        #     policy = policy_alg(net_type, env, policy_kwargs=policy_kwargs, seed=sd, nminibatches=4,
        #                         batch_size=args.bs, noptepochs=args.n_epochs, learning_rate=args.lr)


    # best_mean_reward = -np.inf
    t0 = time.time()
    n_iters = (args.n_train // args.n_train_per_iter) + 1
    for i in range(n_iters):
        mean_reward = simple_eval(policy, eval_env, args.n_test_episodes)
        print(f'agent: {args.env}_side={args.ta_i}_id={args.id}_t={args.m_train}m, step: {i * args.n_train_per_iter}, '
              f'mean_reward: {round(mean_reward, 2)}, time: {round((time.time() - t0) / 3600, 1)} hrs, '
              f'agents: ({args.agent0_ckpt}, {args.agent1_ckpt})')
        # if mean_reward > best_mean_reward:
        #     best_mean_reward = mean_reward
        #     policy.save(args.model_dir + f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_train}m.zip')
        #     print(f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_train}m model saved')
        policy.learn(args.n_train_per_iter)
        sys.stdout.flush()
    policy.save(args.model_dir + f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_train}m.zip')
    print(f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_train}m model saved')
    print(f'finished training {args.env}_side={args.ta_i}_id={args.id}_t={args.m_train}m for {args.n_train} steps :)')
