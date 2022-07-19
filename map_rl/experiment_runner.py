# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# 
# This file is partially derived from Dopamine with the following original 
# copyright note:
# 
# "Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."
"""Module defining classes and helper methods for general agents.

This script overrides two helper methods in Dopamine's `run_experiment.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from map_rl import map_dqn_agent
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent

from dopamine.discrete_domains.run_experiment import Runner, TrainRunner

import gin.tf



@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  print(agent_name)
  print()
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'map_dqn':
    return map_dqn_agent.MapDQNAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_dqn':
    return jax_dqn_agent.JaxDQNAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_quantile':
    return jax_quantile_agent.JaxQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_rainbow':
    return jax_rainbow_agent.JaxRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_implicit_quantile':
    return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))