# Eureka: Human-Level Reward Design via Coding Large Language Models (ICLR 2024) for Autonomous Vehicles gym environment

<div align="center">

[[PDF]](./results/report_Eureka.pdf)
[[Original Publication]](https://eureka-research.github.io/)

[![Python Version](https://img.shields.io/badge/Python-3.8-blue.svg)](https://github.com/eureka-research/Eureka)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
______________________________________________________________________

![](./results/agent_good.gif)
</div>

 On April 30, 2024, at ICLR, a new paper titled “Eureka: Human-Level Reward
 Design via Coding Large Language Models” by Yecheng Jason Ma, William Liang,
 Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi
 “Jim” Fan, and Anima Anandkumar was published, attracting attention from 191 sites
 by October 31, 2024.

 The Eureka algorithm leverages state-of-the-art large language models (LLMs) like
 GPT-4 for zero-shot generation, code-writing, and in-context improvement, applying
 these capabilities to evolutionary optimization over reward code. The generated
 rewards help train agents to acquire complex skills via reinforcement learning,
 outperforming expert human-engineered rewards without any task-specific prompting
 or predefined reward templates.

##  Context and Goal
The authors suggest this algorithm can extend to various environments to
enhance agent performance by generating new reward functions. Our objective was to
validate this claim by applying the Eureka algorithm to a complex stochastic gym
environment, [CarRacing](https://www.gymlibrary.dev/environments/box2d/car_racing/), where the agent’s primary goal is to complete the track as
quickly and steadily as possible.
The CarRacing environment is considered inherently stochastic due to the following
factors:

* Track Variability: The environment generates procedurally different
tracks for each episode, requiring the agent to adapt its driving skills to handle
various terrains and turns instead of memorising a single track.
* Physics and Control Variability: Small changes in physics calculations
(e.g., friction, acceleration) and control precision can yield different
trajectories, especially at high speeds.
 
Designing a reward function for this environment is non-trivial, as the reward depends
on multiple factors, including the agent's speed, position, acceleration, friction, and
steering. Effective reward construction is essential to guide the agent in completing
the track efficiently and stably- something not feasible with manual reward
engineering

## Installation
Eureka requires Python ≥ 3.8. We have tested on Windows 11.

1. Create a new conda environment with:
    ```
    conda create -n eureka python=3.8
    conda activate eureka
    ```

2. Install ollama and llama from [here](https://ollama.com/) 
3. Install all the Eureka requirements by running on the project:
```
pip install -e .
```
   - KNOWN PROBLEM: if you get errors during the wheel builds, you will need to install microsoft visual C++ development tools

4. Install rl-baselines3-zoo dependencies:
```
cd rl-baselines3-zoo; pip install -e .
```

# Getting Started

Navigate to the `eureka` directory and run:
```
python eureka.py env=car_racing
```

Each run will create a timestamp folder in `eureka/outputs` that saves the Eureka log as well as all intermediate reward functions and associated policies.

Other command line parameters can be found in `eureka/cfg/config.yaml`. The list of supported environments can be found in `eureka/cfg/env`.

After the training is completed, you can train an agent with use of the best reward function by pasting it in `./rl-baseline3-zoo/rl_zoo3/car_racing_custom.py` and running the command::
```
   python .\rl-baselines3-zoo\rl_zoo3\train.py --algo ppo --env CarRacingCustomTest-v0  -f .\rl-baselines3-zoo\rl_zoo3\logs/
```

To view the results of the training process: 

```
   python .\rl-baselines3-zoo\rl_zoo3\enjoy.py --algo ppo --env CarRacingCustomTest-v0  -f .\rl-baselines3-zoo\rl_zoo3\logs
```