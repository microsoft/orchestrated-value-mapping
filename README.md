# **Orchestrated Value Mapping**

This repository hosts the code release for the paper ["Orchestrated Value Mapping for Reinforcement Learning"](https://arxiv.org/abs/2203.07171), published at [ICLR 2022][map_rl]. This work was done by [Mehdi Fatemi](https://www.microsoft.com/en-us/research/people/mefatemi) (Microsoft Research) and [Arash Tavakoli](https://atavakol.github.io) (Max Planck Institute for Intelligent Systems).

We release a flexible framework, built upon Dopamine ([Castro et al., 2018][dopamine_paper]), for building and orchestrating various mappings over different reward decomposition schemes. This enables the research community to easily explore the design space that our theory opens up and investigate new convergent families of algorithms.

The code has been developed by [Arash Tavakoli](https://github.com/atavakol).

## [LICENSE](https://github.com/microsoft/orchestrated-value-mapping/blob/master/LICENSE)

## [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
 

## Citing

If you make use of our work, please use the citation information below:

```
@inproceedings{Fatemi2022Orchestrated,
  title={Orchestrated Value Mapping for Reinforcement Learning},
  author={Mehdi Fatemi and Arash Tavakoli},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=c87d0TS4yX}
}
```

# Getting started

We install the required packages within a virtual environment. 


## Virtual environment

Create a virtual environment using `conda` via: 

```
conda create --name maprl-env python=3.8
conda activate maprl-env
```


## Prerequisites

**Atari benchmark.** 
To set up the Atari suite, please follow the steps outlined [here](https://github.com/google/dopamine/blob/master/README.md#prerequisites).    

**Install Dopamine.** Install a compatible version of [Dopamine][dopamine_repo] with `pip`:
```
pip install dopamine-rl==3.1.10
```


## Installing from source

To easily experiment within our framework, install it from source and modify the code directly:

```
git clone https://github.com/microsoft/orchestrated-value-mapping.git
cd orchestrated-value-mapping
pip install -e .
```


## Training an agent

Change directory to the workspace directory:
```
cd map_rl
```

To train a **LogDQN** agent, similar to that introduced by [van Seijen, Fatemi & Tavakoli (2019)][log_rl], run the following command:
```
python -um map_rl.train \
  --base_dir=/tmp/log_dqn \
  --gin_files='configs/map_dqn.gin' \
  --gin_bindings='MapDQNAgent.map_func_id="[log,log]"' \
  --gin_bindings='MapDQNAgent.rew_decomp_id="polar"' &
```
Here, `polar` refers to the reward decomposition scheme described in Equation 13 of [Fatemi & Tavakoli (2022)][map_rl] (which has two reward channels) and `[log,log]` results in a logarithmic mapping for each of the two reward channels. 

Train a **LogLinDQN** agent, similar to that described by [Fatemi & Tavakoli (2022)][map_rl], using:
```
python -um map_rl.train \
  --base_dir=/tmp/loglin_dqn \
  --gin_files='configs/map_dqn.gin' \
  --gin_bindings='MapDQNAgent.map_func_id="[loglin,loglin]"' \
  --gin_bindings='MapDQNAgent.rew_decomp_id="polar"' &
```


## Creating custom agents

To instantiate a custom agent, simply set the mapping functions for each channel and a reward decomposition scheme. For instance, the following setting
```
MapDQNAgent.map_func_id="[log,identity]"
MapDQNAgent.rew_decomp_id="polar"
```
results in a logarithmic mapping for the positive-reward channel and the identity mapping (same as in [DQN][dqn]) for the negative-reward channel. 

To use more complex reward decomposition schemes, such as Configurations 1 and 2 from [Fatemi & Tavakoli (2022)][map_rl], you can do as follows:
```
MapDQNAgent.map_func_id="[identity,identity,log,log,loglin,loglin]"
MapDQNAgent.rew_decomp_id="config_1"
```

To instantiate an ensemble of two learners, each using a `polar` reward decomposition, use the following syntax:
```
MapDQNAgent.map_func_id="[loglin,loglin,log,log]"
MapDQNAgent.rew_decomp_id="two_ensemble_polar"
```


## Custom mappings and reward decomposition schemes

To implement custom mapping functions and reward decomposition schemes, we suggest that you draw on insights from [Fatemi & Tavakoli (2022)][map_rl] and follow the format of such methods in [map_dqn_agent.py](https://github.com/microsoft/orchestrated-value-mapping/map_rl/map_dqn_agent.py) to design yours.  



[map_rl]: https://openreview.net/forum?id=c87d0TS4yX
[log_rl]: https://arxiv.org/abs/1906.00572
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[dopamine_paper]: https://arxiv.org/abs/1812.06110
[dopamine_repo]: https://github.com/google/dopamine