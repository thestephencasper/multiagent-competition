import argparse
import random

POLICY_KWARGS_MLP = {'net_arch': (128, 128)}
POLICY_KWARGS_LSTM = {'net_arch': (128,), 'lstm_hidden_size': 128, 'n_lstm_layers': 1}


def parse_args():

    # some hyperparams https://github.com/HumanCompatibleAI/adversarial-policies/blob/master/src/aprl/train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='')
    parser.add_argument('--agent0_ckpt', type=str, default='')
    parser.add_argument('--agent1_ckpt', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--ta_i', type=int, default=0)  # training agent index, either 0 or 1
    parser.add_argument('--id', type=int, default=0)  # id in order to distinguish identically trained agents
    parser.add_argument('--n_test_episodes', type=int, default=10)
    parser.add_argument('--n_envs', type=int, default=32)
    parser.add_argument('--n_train', type=int, default=int(2e6))
    parser.add_argument('--m_train', type=int, default=2)  # million steps of total training, for saving names
    parser.add_argument('--n_train_per_iter', type=int, default=262144)  # not an alg param, a reporting param
    parser.add_argument('--n_steps', type=int, default=16384)  # how many steps to take per update
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--bs', type=int, default=4096)
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_sb3', type=bool, default=True)
    args = parser.parse_args()
    assert args.ta_i in [0, 1]

    def sample_ckpt(ckpt):
        if 'sample_id' in ckpt:
            ckpt = ckpt.replace('sample_id', random.choice('01'))
        if 'sample_oppt' in ckpt:
            n_each = int(args.n_train // 1e6)
            mx = max(n_each, args.m_train - n_each)
            m_half = int(mx / 2)
            mn = max(n_each, m_half + (m_half % n_each))
            ckpt = ckpt.replace('sample_oppt', str(random.choice(list(range(mn, mx+1, n_each)))))
        return ckpt
    args.agent0_ckpt = sample_ckpt(args.agent0_ckpt)
    args.agent1_ckpt = sample_ckpt(args.agent1_ckpt)

    args.n_steps = int(args.n_steps / args.n_envs)

    return args


