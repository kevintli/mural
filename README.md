# MURAL
Official code for MURAL: Meta-Learning Uncertainty-Aware Rewards for Outcome-Driven RL (ICML 2021)

**MURAL: Meta-Learning Uncertainty-Aware Rewards for Outcome-Driven Reinforcement Learning**\
Kevin Li*, Abhishek Gupta*, Ashwin D Reddy, Vitchyr Pong, Aurick Zhou, Justin Yu, Sergey Levine\
_International Conference on Machine Learning (ICML) 2021_

[Website](sites.google.com/view/mural-rl) | [Paper](https://arxiv.org/abs/2107.07184) 

<img src="https://user-images.githubusercontent.com/6785060/134468427-0d9881df-1cd1-48e3-83d0-7d01a7325bbf.png" data-canonical-src="https://user-images.githubusercontent.com/6785060/134468427-0d9881df-1cd1-48e3-83d0-7d01a7325bbf.png" width="700" />


## Setup Instructions
0. Clone the repository

1. Create a conda environment with the required dependencies, and activate it (2 commands):
```
conda env create -f environment.yml
conda activate mural
```

2. Add the necessary paths (2 commands):
```
pip install -e .
conda develop meta-nml
```

3. Install subfolder dependencies (2 commands):
```
cd meta-nml && pip install -r requirements.txt
cd ../multiworld && pip install -e .
```

4. Enable execution for all run scripts:
```
cd .. && chmod +x scripts/examples/*.sh
```

## Running MURAL
We have included separate scripts for each of the environments in the paper. Use the following commands to run MURAL on the desired environment:
* **Zigzag Maze**: `./scripts/examples/run_zigzag_maze.sh`
* **Spiral Maze**: `./scripts/examples/run_spiral_maze.sh`
* **Sawyer Push**: `./scripts/examples/run_sawyer_push.sh`
* **Sawyer Pick-and-Place**: `./scripts/examples/run_sawyer_pick.sh`
* **Sawyer Door**: `./scripts/examples/run_sawyer_door.sh`
* **Ant Locomotion**: `./scripts/examples/run_ant_maze.sh`
* **Dexterous Hand**: Unfortunately, the code for the Dexterous Hand environment is private and we have been asked not to include it in this submission for the time being.


## Common Issues
**numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject**

Uninstall and reinstall numpy:
```
pip uninstall numpy
pip install numpy
```

**TypeError: __init__() got an unexpected keyword argument 'tags'**

Install an earlier gym version: 
```
pip install gym==0.15.4
```

**Missing aiohttp**
```
pip install aiohttp psutil
```

## Acknowledgements
This codebase was built off of the following publicly available repos:
* **softlearning** (implementation of SAC and other common RL algorithms): https://github.com/rail-berkeley/softlearning
* **multiworld** (multitask gym environments for RL): https://github.com/vitchyr/multiworld
* **pytorch-maml** (implementation of Model-Agnostic Meta-Learning): https://github.com/tristandeleu/pytorch-maml
