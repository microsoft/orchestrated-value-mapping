# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# 
# This file is derived from Dopamine with the following original 
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
"""The standard DQN replay memory, with added support for various reward dtypes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.replay_memory import circular_replay_buffer

import numpy as np

import gin.tf



@gin.configurable
class OutOfGraphTypedReplayBuffer(circular_replay_buffer.OutOfGraphReplayBuffer):
  """Adds support for various reward dtypes in Dopamine's original out-of-graph 
  Replay Buffer."""

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    super(OutOfGraphTypedReplayBuffer, self).__init__(
      observation_shape=observation_shape,
      stack_size=stack_size,
      replay_capacity=replay_capacity,
      batch_size=batch_size,
      update_horizon=update_horizon,
      gamma=gamma,
      max_sample_attempts=max_sample_attempts,
      extra_storage_types=extra_storage_types,
      observation_dtype=observation_dtype,
      terminal_dtype=terminal_dtype,
      action_shape=action_shape,
      action_dtype=action_dtype,
      reward_shape=reward_shape,
      reward_dtype=reward_dtype)
    self._cumulative_discount_vector = np.array(
        [self._gamma**n for n in range(update_horizon)],
        dtype=reward_dtype)


@gin.configurable(
    denylist=['observation_shape', 'stack_size', 'update_horizon', 'gamma'])
class WrappedTypedReplayBuffer(circular_replay_buffer.WrappedReplayBuffer):
  """Wrapper of OutOfGraphTypedReplayBuffer with an in graph sampling mechanism."""

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=False,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    if replay_capacity < update_horizon + 1:
      raise ValueError(
          'Update horizon ({}) should be significantly smaller '
          'than replay capacity ({}).'.format(update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    self.batch_size = batch_size

    # Mainly used to allow subclasses to pass self.memory.
    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = OutOfGraphTypedReplayBuffer(
          observation_shape,
          stack_size,
          replay_capacity,
          batch_size,
          update_horizon,
          gamma,
          max_sample_attempts,
          extra_storage_types=extra_storage_types,
          observation_dtype=observation_dtype,
          terminal_dtype=terminal_dtype,
          action_shape=action_shape,
          action_dtype=action_dtype,
          reward_shape=reward_shape,
          reward_dtype=reward_dtype)

    self.create_sampling_ops(use_staging)