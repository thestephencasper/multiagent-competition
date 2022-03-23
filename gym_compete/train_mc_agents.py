import sys
import gym
from stable_baselines3.ppo import PPO
from gym.envs.registration import register
import os

register(
    id='multicomp/SumoAnts-v1',
    entry_point='gym_compete.new_envs:SumoEnv',
    kwargs={'agent_names': ['ant_fighter', 'ant_fighter'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body_arena.ant_body.ant_body.xml"
            ),
            'world_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets", 'world_body_arena.xml'
            ),
            'init_pos': [(-1, 0, 2.5), (1, 0, 2.5)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5
            },
)

ENV = 'sumo-ants'

if ENV == 'kick-and-defend':
    env = gym.make('multicomp/KickAndDefend-v0')
    policy_type = 'LstmPolicy'
elif ENV == 'run-to-goal-humans':
    env = gym.make('multicomp/RunToGoalHumans-v0')
    policy_type = 'MlpPolicy'
elif ENV == 'run-to-goal-ants':
    env = gym.make('multicomp/RunToGoalAnts-v0')
    policy_type = 'MlpPolicy'
elif ENV == 'you-shall-not-pass':
    env = gym.make('multicomp/YouShallNotPassHumans-v0')
    policy_type = 'MlpPolicy'
elif ENV == 'sumo-humans':
    env = gym.make('multicomp/SumoHumans-v0')
    policy_type = 'LstmPolicy'
elif ENV == 'sumo-ants':
    # env = gym.make('multicomp/SumoAnts-v0')
    env = gym.make('multicomp/SumoAnts-v1')
    policy_type = 'LstmPolicy'
else:
    print('unsupported environment')
    print(
        'choose from: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend')
    sys.exit()

policy = []
for _ in range(2):
    policy.append(PPO(policy_type, env, ent_coef=0.01, policy_kwargs={'net_arch': (128, 128)}))

max_episodes = 50
num_episodes = 0
nstep = 0
total_reward = [0.0 for _ in range(len(policy))]
total_scores = [0 for _ in range(len(policy))]
observation = env.reset()
while num_episodes < max_episodes:
    action = tuple([policy[i].predict(observation=observation[i], deterministic=False) for i in range(len(policy))])
    observation, reward, done, infos = env.step(action)
    nstep += 1
    for i in range(len(policy)):
        total_reward[i] += reward[i]
    if done[0]:
        num_episodes += 1
        draw = True
        for i in range(len(policy)):
            if 'winner' in infos[i]:
                draw = False
                total_scores[i] += 1
                print('Winner: Agent {}, Scores: {}, Total Episodes: {}'.format(i, total_scores, num_episodes))
        if draw:
            print('Game Tied: Agent {}, Scores: {}, Total Episodes: {}'.format(i, total_scores, num_episodes))
        observation = env.reset()
        nstep = 0
        total_reward = [0.0 for _ in range(len(policy))]
        for i in range(len(policy)):
            policy[i].reset()
        print(f'episode: {num_episodes}, total_reward: {total_reward}')
