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
from gym_compete.utils import parse_args, POLICY_KWARGS
from gym_compete.envs import get_env_and_policy, make_env


if __name__ == '__main__':

    freeze_support()
    torch.manual_seed(int(str(time.time()).replace('.', '')[-5:]))
    args = parse_args()
    tac = args.agent1_ckpt if args.ta_i else args.agent0_ckpt  # training agent checkpoint
    fac = args.agent0_ckpt if args.ta_i else args.agent1_ckpt  # fixed agent checkpoint
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = 'cpu'

    env_type, policy_alg = get_env_and_policy(args.env)
    env = DummyVecEnv([make_env(env_type, fac, policy_alg, args, 0)])
    # env = Monitor(make_env(env_type, fac, policy_alg, args, 0)(), './video', force=True)
    net_type = 'MlpPolicy' if policy_alg == PPO else 'MlpLstmPolicy'
    policy = policy_alg(net_type, env, ent_coef=args.ent_coef, policy_kwargs=POLICY_KWARGS,
                        batch_size=args.batch_size, n_epochs=args.n_epochs, device=device)
    policy.set_parameters(load_path_or_dict=(args.model_dir + tac), device=device)

    video_folder = './video/'
    video_len = 100
    _ = env.reset()
    env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0, video_length=video_len,
                           name_prefix=args.agent0_ckpt + '_vs_' + args.agent1_ckpt)
    obs = env.unwrapped.reset()
    for _ in range(video_len + 1):
        action = policy.predict(observation=obs, deterministic=False)[0]
        obs, _, _, _ = env.step(action)
    env.close()

    # # this works
    # # env_id = 'CartPole-v1'
    # env_id = 'Ant-v2'
    # video_folder = './video/'
    # video_length = 100
    # env = DummyVecEnv([lambda: gym.make(env_id)])
    # obs = env.reset()
    # env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x == 0,
    #                        video_length=video_length, name_prefix=env_id)
    # _ = env.reset()
    # for _ in range(video_length + 1):
    #     action = [env.action_space.sample()]
    #     _, _, _, _ = env.step(action)
    # env.close()

    # # this works!
    # # env_id = 'CartPole-v1'
    # env_id = 'Ant-v2'
    # env = Monitor(gym.make(env_id), './video', force=True)
    # obs = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     _, _, done, _ = env.step(action)
    # env.close()

    # # this does not work
    # # env_id = 'CartPole-v1'
    # env_id = 'Ant-v2'
    # env = gym.make(env_id)
    # vr = VideoRecorder(env, './video/tmp.mp4', enabled=True)
    # obs = env.reset()
    # done = False
    # while not done:
    #     env.unwrapped.render()  # env.unwrapped.render()
    #     vr.capture_frame()
    #     action = env.action_space.sample()
    #     _, _, done, _ = env.step(action)
    # vr.close()
    # vr.enabled = False
    # env.close()

    # # this works
    # # env_id = 'CartPole-v1'
    # env_id = 'Ant-v2'
    # env = gym.make(env_id)
    # obs = env.reset()
    # done = False
    # while not done:
    #     frame = env.render(mode="rgb_array")  # env.unwrapped.render()
    #     print(frame)
    #     action = env.action_space.sample()
    #     _, _, done, _ = env.step(action)
    # env.close()
    # print(':)')

