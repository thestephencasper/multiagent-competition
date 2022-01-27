# Competitive Multi-Agent Environments - Updated

This repository contains the environments for the paper [Emergent Complexity via Multi-agent Competition](https://arxiv.org/abs/1710.03748)

I updated the fork of the original code to be compatible with MuJoCo 2.1 and hopefully newer versions in the future.  
I removed policy.py, which is not needed if you just use the gym_compete environments.
If you need that module, look at v0.1.0 of this fork, which has a version that was updated to work with tf2, by using my WIP stable-baselines fork which has added support for tf2 (https://github.com/PavelCz/stable-baselines-tf2.git).

## Dependencies
Use `pip install -r requirements.txt` to install dependencies. If you haven't used MuJoCo before, please refer to the [installation guide](https://github.com/openai/mujoco-py).
The code has been tested with the following dependencies:
* Python version 3.7
* [OpenAI GYM](https://github.com/openai/gym) version 0.19 with MuJoCo 2.1 support (use mujoco-py version 2.1)
* [Tensorflow](https://www.tensorflow.org/versions/r1.1/install/) version 2.7.0
* [Numpy](https://scipy.org/install.html) version 1.21.5

## Installing Package
After installing all dependencies, make sure gym works with support for MuJoCo environments.
Next install `gym-compete` package as:
```bash
cd gym-compete
pip install -e .
```

Or use pip to install / add to requirements as
```bash
gym_compete @ git+https://github.com/PavelCz/multiagent-competition.git@v0.1.0
```
Check install is successful by coming out of the directory and trying `import gym_compete` in python console. Some users might require a `sudo pip install`.

## Trying the environments
Agent policies are provided for the various environments in folder `agent-zoo`. To see a demo of all the environments do:
```bash
bash demo_tasks.sh all
```
To instead try a single environment use:
```bash
bash demo_tasks.sh <task>
```
where `<task>` is one of: `run-to-goal-humans`, `run-to-goal-ants`, `you-shall-not-pass`, `sumo-ants`, `sumo-humans` and `kick-and-defend`
