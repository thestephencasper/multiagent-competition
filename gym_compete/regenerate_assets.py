from gym.envs import registry
import gym_compete  # for side-effects


def main():
    for env_spec in registry.all():
        if env_spec.id.startswith('multicomp/'):
            env_spec.make()


if __name__ == '__main__':
    main()