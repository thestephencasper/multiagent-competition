import sys
import os
import copy
import argparse
import time
import gym
import multiprocessing
from multiprocessing import freeze_support
import numpy as np
import torch
import gym_compete
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, StackedObservations
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath('..'))

POLICY_ALG = PPO
POLICY_KWARGS = {'net_arch': (128, 128)}
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda:0'


def parse_args():

    # some hyperparams https://github.com/HumanCompatibleAI/adversarial-policies/blob/master/src/aprl/train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='kick-and-defend')
    parser.add_argument('--agent0_ckpt', type=str, default='')
    parser.add_argument('--agent1_ckpt', type=str, default='')
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--save_dir', type=str, default='./models/')
    parser.add_argument('--n_stack', type=int, default=5)  # training agent index, either 0 or 1
    parser.add_argument('--ta_i', type=int, default=0)  # training agent index, either 0 or 1
    parser.add_argument('--n_test_episodes', type=int, default=50)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=int(1e8))
    parser.add_argument('--n_steps_per_iter', type=int, default=int(2**15))
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=int(2**13))
    parser.add_argument('--n_epochs', type=int, default=5)
    args = parser.parse_args()
    assert args.ta_i in [0, 1]
    return args


class EnvWrapper:
    # for training one agent as opposed to two
    def __init__(self, env, ta_i, fixed_agent_checkpoint, args):
        self.env = env
        self.metadata = self.env.metadata
        self.ta_i = ta_i
        self.observation_space = self.env.observation_space[self.ta_i]
        self.action_space = self.env.action_space[self.ta_i]
        self.observations = None

        if fixed_agent_checkpoint:
            self.fixed_agent = POLICY_ALG.load(fixed_agent_checkpoint)
        else:
            eos = copy.copy(self.env.observation_space)
            eas = copy.copy(self.env.action_space)
            self.env.observation_space = eos[1-self.ta_i]
            self.env.action_space = eas[1-self.ta_i]
            self.fixed_agent = POLICY_ALG(args.policy_type, self.env, ent_coef=args.ent_coef, policy_kwargs=POLICY_KWARGS)
            self.env.observation_space = eos
            self.env.action_space = eas

    def step(self, action):
        fixed_action = self.fixed_agent.predict(observation=self.observations[1-self.ta_i], deterministic=False)[0]
        action = (fixed_action, action) if self.ta_i else (action, fixed_action)
        self.observations, reward, done, infos = self.env.step(action)
        return self.observations[self.ta_i], reward[self.ta_i], done, infos[self.ta_i]

    def reset(self):
        self.observations = self.env.reset()
        return self.observations[self.ta_i]


def make_env(env_id, fac, ta_i, args, rank, seed=0):
    # for subproc vec env
    def _init():
        env = EnvWrapper(gym.make(env_id), ta_i, fac, args)
        env.env.seed(seed+rank)
        return env
    set_random_seed(seed)
    return _init


def simple_eval(policy, eval_env, n_episodes):
    all_rewards = []
    total_wins = 0
    observation = eval_env.reset()
    for _ in range(n_episodes):
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.predict(observation=observation, deterministic=False)[0]
            observation, reward, done, infos = eval_env.step(action)
            done = done[0]
            ep_reward += reward[0]
        all_rewards.append(ep_reward)
        if 'winner' in infos[0]:
            total_wins += 1
        observation = eval_env.reset()
    return sum(all_rewards) / n_episodes


def get_env_type_and_to_stack(env_key):
    
    if env_key == 'kick-and-defend':
        # asymmetric
        env_type = 'multicomp/KickAndDefend-v0'
        to_stack = True
    elif env_key == 'run-to-goal-humans':
        # symmetric, not doen by Gleave
        env_type = 'multicomp/RunToGoalHumans-v0'
        to_stack = False
    elif env_key == 'run-to-goal-ants':
        # symmetric, not doen by Gleave
        env_type = 'multicomp/RunToGoalAnts-v0'
        to_stack = False
    elif env_key == 'you-shall-not-pass':
        # asymmetric
        env_type = 'multicomp/YouShallNotPassHumans-v0'
        to_stack = False
    elif env_key == 'sumo-humans':
        # symmetric
        env_type = 'multicomp/SumoHumans-v0'
        to_stack = True
    elif env_key == 'sumo-ants':
        # symmetric
        env_type = 'multicomp/SumoAnts-v0'
        to_stack = True
    else:
        print('unsupported environment')
        print('must be: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend')
        sys.exit()
    return env_type, to_stack


if __name__ == '__main__':

    freeze_support()
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    tac = args.agent1_ckpt if args.ta_i else args.agent0_ckpt  # training agent checkpoint
    fac = args.agent0_ckpt if args.ta_i else args.agent1_ckpt  # fixed agent checkpoint

    env_type, to_stack = get_env_type_and_to_stack(args.env)
    env = SubprocVecEnv([make_env(env_type, fac, args.ta_i, args, i) for i in range(args.n_envs)])
    eval_env = SubprocVecEnv([make_env(env_type, fac, args.ta_i, args, 42)])
    if to_stack:
        env = VecFrameStack(env, args.n_stack)
        eval_env = VecFrameStack(eval_env, args.n_stack)
    if tac:  # trained agent checkpoint
        policy = POLICY_ALG.load(tac)
    else:
        policy = POLICY_ALG(args.policy_type, env, ent_coef=args.ent_coef, policy_kwargs=POLICY_KWARGS,
                            batch_size=args.batch_size, n_epochs=args.n_epochs, device=DEVICE)

    best_mean_reward = -np.inf
    t0 = time.time()
    n_iters = (args.n_steps // args.n_steps_per_iter) + 1
    for i in range(n_iters):
        mean_reward = simple_eval(policy, eval_env, args.n_test_episodes)
        print(f'{args.env}, {args.ta_i}, step: {i * args.n_steps_per_iter}, mean_reward: {round(mean_reward, 2)}, time: {round(time.time() - t0)}')
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            policy.save(args.save_dir + 'tmp.sb')
            print('model saved')
        policy.learn(args.n_steps_per_iter)
        sys.stdout.flush() 

