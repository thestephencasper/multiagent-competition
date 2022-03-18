# Competitive Multi-Agent Environments - MuJoCo 2.1

This repository contains the environments for the paper [Emergent Complexity via Multi-agent Competition](https://arxiv.org/abs/1710.03748)

This fork is updated to be compatible with MuJoCo 2.1.
I removed `policy.py` as I'm not using it and it relies on updating stable-baselines to tf2.
Version v0.1.0 of this fork, supports `policy.py` using [this fork](https://github.com/PavelCz/stable-baselines-tf2.git) of sb, to upgrade it to tf2.

## Dependencies
Use `pip install -r requirements.txt` to install dependencies. If you haven't used MuJoCo before, please refer to the [installation guide](https://github.com/openai/mujoco-py).
The code has been tested with the following dependencies:
* Python version 3.8
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
gym_compete @ git+https://github.com/PavelCz/multiagent-competition.git@v0.2.1
```
Check install is successful by coming out of the directory and trying `import gym_compete` in python console. Some users might require a `sudo pip install`.

