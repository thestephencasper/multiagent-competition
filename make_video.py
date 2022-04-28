import os
import copy
import time
import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from multiprocessing import freeze_support
import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from gym_compete.utils import parse_args, POLICY_KWARGS_MLP, POLICY_KWARGS_LSTM
from gym_compete.envs import get_env_and_policy, make_env


if __name__ == '__main__':

    freeze_support()
    torch.manual_seed(int(str(time.time()).replace('.', '')[-5:]))
    args = parse_args()
    tac = args.agent1_ckpt if args.ta_i else args.agent0_ckpt  # training agent checkpoint
    fac = args.agent0_ckpt if args.ta_i else args.agent1_ckpt  # fixed agent checkpoint
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = 'cpu'

    env_type, policy_alg, policy_kwargs, net_type = get_env_and_policy(args.env, args.use_sb3)
    env = make_env(env_type, fac, policy_alg, net_type, policy_kwargs, args, 0)()
    savename = f'{args.agent0_ckpt}_vs_{args.agent1_ckpt}'
    vr = VideoRecorder(env, f'./video/{savename}.mp4', enabled=True)

    if args.use_sb3:
        policy = policy_alg(net_type, env, policy_kwargs=policy_kwargs)
        if tac:
            policy.set_parameters(load_path_or_dict=(args.model_dir + tac))
    else:
        raise NotImplementedError
        # if tac:
        #     policy = policy_alg.load(args.model_dir + tac)
        # else:
        #     policy = policy_alg(net_type, env, policy_kwargs=policy_kwargs)

    video_folder = './video/'
    obs = env.reset()
    for _ in range(1000 + 1):
        vr.capture_frame()
        action = policy.predict(observation=obs, deterministic=False)[0]
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    vr.close()
    env.close()

    print(f'saved {savename} :)')

