import argparse

POLICY_KWARGS = {'net_arch': (128, 128)}


def parse_args():

    # some hyperparams https://github.com/HumanCompatibleAI/adversarial-policies/blob/master/src/aprl/train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='')
    parser.add_argument('--agent0_ckpt', type=str, default='')
    parser.add_argument('--agent1_ckpt', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--ta_i', type=int, default=0)  # training agent index, either 0 or 1
    parser.add_argument('--id', type=int, default=0)  # id in order to distinguish identically trained agents
    parser.add_argument('--n_test_episodes', type=int, default=50)
    parser.add_argument('--n_envs', type=int, default=6)
    parser.add_argument('--n_steps', type=int, default=int(2e7))
    parser.add_argument('--m_steps', type=int, default=20)  # million steps of total training, for saving names
    parser.add_argument('--n_steps_per_iter', type=int, default=int(2**18))
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=int(2**12))
    parser.add_argument('--n_epochs', type=int, default=5)
    args = parser.parse_args()
    assert args.ta_i in [0, 1]
    return args


