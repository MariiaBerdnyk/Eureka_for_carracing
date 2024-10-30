# Eureka: Human-Level Reward Design via Coding Large Language Models (ICLR 2024)

<div align="center">

[[Website]](https://eureka-research.github.io)
[[arXiv]](https://arxiv.org/abs/2310.12931)
[[PDF]](https://eureka-research.github.io/assets/eureka_paper.pdf)

[![Python Version](https://img.shields.io/badge/Python-3.8-blue.svg)](https://github.com/eureka-research/Eureka)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/eureka-research/Eureka)](https://github.com/eureka-research/Eureka/blob/main/LICENSE)
______________________________________________________________________

https://github.com/eureka-research/Eureka/assets/21993118/1abb960d-321a-4de9-b311-113b5fc53d4a



![](images/eureka.png)
</div>

Large Language Models (LLMs) have excelled as high-level semantic planners for sequential decision-making tasks. However, harnessing them to learn complex low-level manipulation tasks, such as dexterous pen spinning, remains an open problem. We bridge this fundamental gap and present Eureka, a **human-level** reward design algorithm powered by LLMs. Eureka exploits the remarkable zero-shot generation, code-writing, and in-context improvement capabilities of state-of-the-art LLMs, such as GPT-4, to perform in-context evolutionary optimization over reward code. The resulting rewards can then be used to acquire complex skills via reinforcement learning. Eureka generates reward functions that outperform expert human-engineered rewards without any task-specific prompting or pre-defined reward templates. In a diverse suite of 29 open-source RL environments that include 10 distinct robot morphologies, Eureka outperforms human expert on **83\%** of the tasks leading to an average normalized improvement of **52\%**. The generality of Eureka also enables a new gradient-free approach to reinforcement learning from human feedback (RLHF), readily incorporating human oversight to improve the quality and the safety of the generated rewards in context. Finally, using Eureka rewards in a curriculum learning setting, we demonstrate for the first time a simulated five-finger Shadow Hand capable of performing pen spinning tricks, adeptly manipulating a pen in circles at human speed. 

# Installation
Eureka requires Python ≥ 3.8. We have tested on Ubuntu 20.04 and 22.04.

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