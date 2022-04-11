import sys
import os
import time
import multiprocessing
from multiprocessing import freeze_support
import numpy as np
import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_compete.utils import parse_args, POLICY_KWARGS
from gym_compete.envs import make_env, get_env_and_policy, simple_eval


if __name__ == '__main__':

    freeze_support()
    multiprocessing.set_start_method('spawn')
    torch.manual_seed(int(str(time.time()).replace('.', '')[-5:]))
    args = parse_args()
    tac = args.agent1_ckpt if args.ta_i else args.agent0_ckpt  # training agent checkpoint
    fac = args.agent0_ckpt if args.ta_i else args.agent1_ckpt  # fixed agent checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ta_i)

    env_type, policy_alg = get_env_and_policy(args.env)
    env = SubprocVecEnv([make_env(env_type, fac, policy_alg, args, i) for i in range(args.n_envs)])
    eval_env = SubprocVecEnv([make_env(env_type, fac, policy_alg, args, 42)])
    net_type = 'MlpPolicy' if policy_alg == PPO else 'MlpLstmPolicy'
    policy = policy_alg(net_type, env, ent_coef=args.ent_coef, policy_kwargs=POLICY_KWARGS,
                        batch_size=args.batch_size, n_epochs=args.n_epochs, device=device)
    if tac:
        policy.set_parameters(load_path_or_dict=(args.model_dir + tac), device=device)

    best_mean_reward = -np.inf
    t0 = time.time()
    n_iters = (args.n_steps // args.n_steps_per_iter) + 1
    for i in range(n_iters):
        mean_reward = simple_eval(policy, eval_env, args.n_test_episodes)
        print(
            f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_steps}m, step: {i * args.n_steps_per_iter}, mean_reward: {round(mean_reward, 2)}, time: {round((time.time() - t0) / 3600)} hrs')
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            policy.save(args.model_dir + f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_steps}m.sb')
            print(f'{args.env}_side={args.ta_i}_id={args.id}_t={args.m_steps}m model saved')
        policy.learn(args.n_steps_per_iter)
        sys.stdout.flush()
    print(f'finished training {args.env}_side={args.ta_i}_id={args.id}_t={args.m_steps}m for {args.n_steps} steps :)')
