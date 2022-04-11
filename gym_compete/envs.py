import sys
import copy
import gym
from stable_baselines3.ppo import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import set_random_seed
from gym_compete.utils import POLICY_KWARGS

# sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath('..'))


class EnvWrapper:
    # for training one agent as opposed to two
    def __init__(self, env, ta_i, fixed_agent_checkpoint, policy_alg, args):
        self.env = env
        self.metadata = self.env.metadata
        self.ta_i = ta_i
        self.observation_space = self.env.observation_space[self.ta_i]
        self.action_space = self.env.action_space[self.ta_i]
        self.observations = None
        self.spec = self.env.spec

        if fixed_agent_checkpoint:
            self.fixed_agent = policy_alg.load(args.model_dir + fixed_agent_checkpoint)
        else:
            eos = copy.copy(self.env.observation_space)
            eas = copy.copy(self.env.action_space)
            self.env.observation_space = eos[1 - self.ta_i]
            self.env.action_space = eas[1 - self.ta_i]
            net_type = 'MlpPolicy' if policy_alg == PPO else 'MlpLstmPolicy'
            self.fixed_agent = policy_alg(net_type, self.env, ent_coef=args.ent_coef, policy_kwargs=POLICY_KWARGS)
            self.env.observation_space = eos
            self.env.action_space = eas

    def step(self, action):
        fixed_action = self.fixed_agent.predict(observation=self.observations[1 - self.ta_i], deterministic=False)[0]
        action = (fixed_action, action) if self.ta_i else (action, fixed_action)
        self.observations, reward, done, infos = self.env.step(action)
        return self.observations[self.ta_i], reward[self.ta_i], done, infos[self.ta_i]

    def reset(self):
        self.observations = self.env.reset()
        return self.observations[self.ta_i]

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def make_env(env_id, fac, policy_alg, args, rank, seed=0):
    # for subproc vec env
    def _init():
        env = EnvWrapper(gym.make(env_id), args.ta_i, fac, policy_alg, args)
        env.env.seed(seed + rank)
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


def get_env_and_policy(env_key):
    if env_key == 'kick-and-defend':
        # asymmetric
        env_type = 'multicomp/KickAndDefend-v0'
        policy = RecurrentPPO
    # elif env_key == 'run-to-goal-humans':
    #     # symmetric, not done by Gleave
    #     env_type = 'multicomp/RunToGoalHumans-v0'
    #     policy = PPO
    # elif env_key == 'run-to-goal-ants':
    #     # symmetric, not done by Gleave
    #     env_type = 'multicomp/RunToGoalAnts-v0'
    #     policy = PPO
    elif env_key == 'you-shall-not-pass':
        # asymmetric
        env_type = 'multicomp/YouShallNotPassHumans-v0'
        policy = PPO
    elif env_key == 'sumo-humans':
        # symmetric
        env_type = 'multicomp/SumoHumans-v0'
        policy = RecurrentPPO
    elif env_key == 'sumo-ants':
        # symmetric
        env_type = 'multicomp/SumoAnts-v0'
        policy = RecurrentPPO
    else:
        print('unsupported environment')
        print(
            'must be: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend')
        sys.exit()
    return env_type, policy
