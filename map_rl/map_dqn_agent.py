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
"""Compact implementation of the MapDQN agent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib

from map_rl import models
from map_rl import circular_replay_buffer

import numpy as np
import tensorflow as tf

import gin.tf



@gin.configurable
class MapDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the MapDQN agent class."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=models.MapDQNNetwork,
               gamma=0.96,
               map_func_id='[loglin,loglin]',
               rew_decomp_id='polar',
               use_gradscaling=False,
               use_nonlinear_heads=False,
               clip_qt_max=True,
               clip_q_chosen=True,
               alpha=0.00025,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               eval_mode=False,
               use_staging=False,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.compat.v1.train.RMSPropOptimizer(
                   learning_rate=0.0025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    assert isinstance(observation_shape, tuple)
    
    try:
      map_func_id = map_func_id.replace(" ", "")
      map_func_id = map_func_id.strip('][').split(',')
    except: 
      pass
    assert isinstance(map_func_id, list) 
    
    try: lr = optimizer._learning_rate
    except: lr = optimizer._lr
    
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t network: %s', network)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t map_func_id: %s', map_func_id)
    logging.info('\t rew_decomp_id: %s', rew_decomp_id)
    logging.info('\t use_gradscaling: %s', use_gradscaling)
    logging.info('\t use_nonlinear_heads: %s', use_nonlinear_heads)
    logging.info('\t clip_qt_max: %s', clip_qt_max)
    logging.info('\t clip_q_chosen: %s', clip_q_chosen)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t tf_device: %s', tf_device)
    logging.info('\t use_staging: %s', use_staging)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t beta_f: %f', lr)
    logging.info('\t beta_reg: %f', alpha / lr)
    logging.info('\t alpha: %f', alpha)
    logging.info('\t max_tf_checkpoints_to_keep: %d', max_tf_checkpoints_to_keep)

    self.tf_float = tf.float64
    self.np_float = np.float64

    self.zero        = tf.constant(0, dtype=self.tf_float)
    self.half        = tf.constant(0.5, dtype=self.tf_float)
    self.one         = tf.constant(1, dtype=self.tf_float)
    self.ten         = tf.constant(10, dtype=self.tf_float)
    self.hundred     = tf.constant(100, dtype=self.tf_float)
    self.thousand    = tf.constant(1000, dtype=self.tf_float)
    self.nine        = tf.constant(9, dtype=self.tf_float)
    self.ninety      = tf.constant(90, dtype=self.tf_float)
    self.ninehundred = tf.constant(900, dtype=self.tf_float)

    self.num_actions = num_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = self.np_float(gamma)
    self.map_func_id = map_func_id
    self.rew_decomp_id = rew_decomp_id
    self.use_gradscaling = use_gradscaling
    self.use_nonlinear_heads = use_nonlinear_heads
    self.clip_qt_max = clip_qt_max
    self.clip_q_chosen = clip_q_chosen
    self.beta_reg = alpha / lr  
    self.update_horizon = update_horizon
    self.cumulative_gamma = self.gamma**self.np_float(update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    if self.rew_decomp_id == 'polar':
      self.num_reward_channels = 2
      lambdas_on_heads = [1,-1]
    elif self.rew_decomp_id == 'config_1':
      self.num_reward_channels = len(self.map_func_id)
      lambdas_on_heads =      [ 10**i for i in range(0, int(self.num_reward_channels / 2))]
      lambdas_on_heads.extend([-10**i for i in range(0, int(self.num_reward_channels / 2))])
    elif self.rew_decomp_id == 'config_2':
      self.num_reward_channels = len(self.map_func_id)
      lambdas_on_heads =      [ 1 if i==0 else  9*(10**(i-1)) for i in range(0, int(self.num_reward_channels / 2))]
      lambdas_on_heads.extend([-1 if i==0 else -9*(10**(i-1)) for i in range(0, int(self.num_reward_channels / 2))])
    elif self.rew_decomp_id == 'two_ensemble_polar':
      self.num_reward_channels = 4
      lambdas_on_heads = [0.5, -0.5, 0.5, -0.5]
    else: 
      raise NotImplementedError(self.rew_decomp_id)

    self.lambdas_on_heads = [self.np_float(head_lambda) for head_lambda in lambdas_on_heads]

    logging.info('\t num_reward_channels: %s', self.num_reward_channels)
    logging.info('\t lambdas_on_heads: %s', self.lambdas_on_heads)
    
    # Logarithmic mapping specific parameters.
    # `Delta` refers to `d` in Eq. 6 of van Seijen, Fatemi, Tavakoli (2019). 
    Delta = self.np_float(0.0)  
    # `(c,log_eps)` refer to `(c,d)` in Sec. 5 of Fatemi & Tavakoli (2022).
    c = self.np_float(0.5)
    self.log_eps = self.np_float(0.01687)  # roughly 0.96**100
    
    self.map_func_params = []
    self.initializers_on_heads = {'kernel_initializers': [],
                                  'bias_initializers':   []}
    for head_map in self.map_func_id:
      assert head_map in ['identity','log','loglin'] 
      self.map_func_params.append([Delta, c])
      self.initializers_on_heads['kernel_initializers'].append('glorot_uniform')
      self.initializers_on_heads['bias_initializers'].append('zeros')

    with tf.device(tf_device):
      # Create a placeholder for the state input to the MapDQN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)

      if self.rew_decomp_id == 'polar':
        self.reward_func = self.polar_decomp
      elif self.rew_decomp_id == 'config_1':
        self.reward_func = self.config_1_decomp
      elif self.rew_decomp_id == 'config_2':
        self.reward_func = self.config_2_decomp
      elif self.rew_decomp_id == 'two_ensemble_polar':
        self.reward_func = self.two_ensemble_polar_decomp
      else: 
        raise NotImplementedError
      
      self.forward_map_funcs = []
      self.inverse_map_funcs = []
      for head_map in self.map_func_id:
        if head_map == 'identity':
          self.forward_map_funcs.append(self.forward_identity_func)
          self.inverse_map_funcs.append(self.inverse_identity_func)
        elif head_map == 'log':
          self.forward_map_funcs.append(self.forward_log_func) 
          self.inverse_map_funcs.append(self.inverse_log_func)
        elif head_map == 'loglin':
          self.forward_map_funcs.append(self.forward_loglin_func)
          self.inverse_map_funcs.append(self.inverse_loglin_func)
        else:
          raise NotImplementedError

      self._build_networks()

      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.compat.v1.summary.merge_all()
    self._sess = sess

    var_map = atari_lib.maybe_transform_variable_names(
        tf.compat.v1.global_variables())
    self._saver = tf.compat.v1.train.Saver(
        var_list=var_map, max_to_keep=max_tf_checkpoints_to_keep)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(
        self.num_actions, 
        self.lambdas_on_heads, 
        self.inverse_map_funcs, 
        self.map_func_params, 
        self.initializers_on_heads, 
        use_gradscaling=self.use_gradscaling, 
        use_nonlinear_heads=self.use_nonlinear_heads,
        tf_float=self.tf_float, name=name)
    return network

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    # Gets greedy actions over the aggregated target-network's Q-values for the 
    # replay's next states, used for retrieving the target Q-values for all heads. 
    self._replay_next_target_net_q_argmax = tf.argmax(
        self._replay_next_target_net_outputs.q_values, axis=1)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperTypedReplayBuffer object.
    """
    return circular_replay_buffer.WrappedTypedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype,
        reward_dtype=np.float64)

  def _build_target_q_op(self):
    """Build an op used as a target for the mapped Q-value.

    Returns:
      target_q_op: An op calculating the mapped Q-value.
    """
    if self.clip_qt_max or self.clip_q_chosen:
      min_return = self.zero
      max_return = self.one / (self.one - self.gamma)
    # One-hot encode the greedy actions over the target-network's aggregated 
    # Q-values for the replay's next states. 
    replay_next_target_net_q_argmax_one_hot = tf.one_hot(
        self._replay_next_target_net_q_argmax, self.num_actions, self.one, self.zero, 
        name='replay_next_target_net_q_argmax_one_hot')
    # One-hot encode the selected actions on the replay's states.
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, self.one, self.zero, 
        name='replay_action_one_hot')
    # Calculate each head's target Q-value (in standard space) with the 
    # action that maximizes the target-network's aggregated Q-values for 
    # the replay's next states.
    map_target_on_heads = []
    for hid, (head_replay_next_qt_values, head_replay_q_values) in enumerate(
        zip(self._replay_next_target_net_outputs.q_values_on_heads,
            self._replay_net_outputs.q_values_on_heads)):
      head_replay_next_qt_max_unclipped = tf.reduce_sum(
          head_replay_next_qt_values * replay_next_target_net_q_argmax_one_hot,
          axis=1, name='head_replay_next_qt_max_unclipped_'+str(hid))

      # Clips the maximum target-network's Q-values across heads 
      # for the replay's next states.
      if self.clip_qt_max:
        head_replay_next_qt_max_clipped_min = tf.maximum(min_return, 
          head_replay_next_qt_max_unclipped)
        head_replay_next_qt_max = tf.minimum(max_return, 
          head_replay_next_qt_max_clipped_min)
      else:
        head_replay_next_qt_max = head_replay_next_qt_max_unclipped

      # Terminal state masking.
      head_replay_next_qt_max_masked = head_replay_next_qt_max * \
          (1. - tf.cast(self._replay.terminals, self.tf_float))
      
      # Creates each head's separate reward signals
      # and bootstraps from the appropriate target for each head.
      head_standard_td_target = self.reward_func(self._replay.rewards, hid) 
      head_standard_td_target += \
          self.cumulative_gamma * head_replay_next_qt_max_masked
      
      # Gets the current-network's head-wise Q-values (in standard 
      # space) for the replay's chosen actions.
      head_replay_chosen_q = tf.reduce_sum(
          head_replay_q_values * replay_action_one_hot,
          axis=1, name='head_replay_chosen_q_'+str(hid))   
      
      if self.clip_q_chosen:
        head_replay_chosen_q = tf.maximum(min_return, 
          head_replay_chosen_q)
        head_replay_chosen_q = tf.minimum(max_return, 
          head_replay_chosen_q)
      
      # Averaging samples in the standard space.
      head_UT_new = head_replay_chosen_q + \
          self.beta_reg * (head_standard_td_target - head_replay_chosen_q)    

      # Clips the minimum TD-targets in the standard space for each head 
      # so as to avoid log(x <= 0) or too close of a positive x to zero.
      if self.map_func_id[hid] == 'log' or self.map_func_id[hid] == 'loglin':
        head_UT_new = tf.maximum(self.log_eps, head_UT_new)

      # Forward mapping.
      head_map_target = self.forward_map_funcs[hid](head_UT_new, *self.map_func_params[hid])

      map_target_on_heads.append(tf.cast(head_map_target, tf.float32))
    return map_target_on_heads

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, self.one, self.zero, name='action_one_hot')
    # Gets the mapping-space target for all heads.
    map_target_on_heads = []
    map_target_on_heads_raw = self._build_target_q_op()
    for head_map_target_raw in map_target_on_heads_raw:
      map_target_on_heads.append(tf.stop_gradient(head_map_target_raw))
    # For the replay's chosen actions, these are the current-network's Q-tilde 
    # values across heads, which will be updated for each head separately.
    loss = 0.
    for hid, (head_q_tilde_values, head_map_target) in enumerate(
        zip(self._replay_net_outputs.q_tilde_values_on_heads, map_target_on_heads)): 
      head_replay_chosen_q_tilde = tf.reduce_sum(
          head_q_tilde_values * replay_action_one_hot, 
          axis=1, name='head_replay_chosen_q_tilde_'+str(hid))
      head_replay_chosen_q_tilde = tf.cast(head_replay_chosen_q_tilde, tf.float32)
      loss += tf.compat.v1.losses.huber_loss(head_map_target, 
          head_replay_chosen_q_tilde, reduction=tf.losses.Reduction.NONE)

    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))

  """ Mapping Functions """

  @tf.function
  def forward_identity_func(self, q_values, Delta, c):
    """Forward mapping following the identity function."""
    return q_values

  @tf.function
  def inverse_identity_func(self, q_tilde_values, Delta, c):
    """Inverse mapping following the identity function."""
    return q_tilde_values

  @tf.function
  def forward_log_func(self, q_values, Delta, c):
    """Forward mapping following a Logarithmic function."""
    return c * tf.math.log(q_values) + Delta

  @tf.function
  def inverse_log_func(self, q_tilde_values, Delta, c):
    """Inverse mapping following a Logarithmic function."""
    return tf.math.exp((q_tilde_values - Delta) / c)

  @tf.function
  def forward_loglin_func(self, q_values, Delta, c):
    """Forward mapping following a piecewise Logarithmic-Linear function."""
    # If `condition` is True, linear mapping is used. 
    # True if q_values > break-point else False.
    condition = tf.greater(q_values, tf.math.exp(-Delta / c))
    log_q_tilde_values = c * tf.math.log(q_values) + Delta
    lin_q_tilde_values = c * tf.math.exp(Delta / c) * q_values - c
    return tf.where(condition, lin_q_tilde_values, log_q_tilde_values)

  @tf.function
  def inverse_loglin_func(self, q_tilde_values, Delta, c):
    """Inverse mapping following a piecewise Logarithmic-Linear function."""
    # If `condition` is True, linear inverse-mapping is used.
    # True if q_values > 0 else False.
    condition = tf.greater(q_tilde_values, 0.)
    log_inv_q_values = tf.math.exp((q_tilde_values - Delta) / c)
    lin_inv_q_values = (q_tilde_values + c) * (tf.math.exp(-Delta / c) / c)
    return tf.where(condition, lin_inv_q_values, log_inv_q_values)

  """ Reward Decomposers """

  @tf.function
  def rewards_above_a(self, replay_rewards, a):
    """Returns rewards that are above `a` or else zero."""
    return replay_rewards * tf.cast(tf.greater(replay_rewards, a), self.tf_float)

  @tf.function
  def rewards_below_a(self, replay_rewards, a):
    """Returns rewards that are below `a` or else zero."""
    return replay_rewards * tf.cast(tf.less(replay_rewards, a), self.tf_float)
  
  @tf.function
  def rewards_between_a_and_b(self, replay_rewards, a, b, equal_a, equal_b):
    """Returns rewards that are between `a` and `b` or else zero."""
    if equal_a: lower_tensor = tf.math.greater_equal(replay_rewards, a)
    else: lower_tensor = tf.math.greater(replay_rewards, a)
    
    if equal_b: upper_tensor = tf.math.less_equal(replay_rewards, b)
    else: upper_tensor = tf.math.less(replay_rewards, b)
    
    in_range = tf.math.logical_and(lower_tensor, upper_tensor)
    return replay_rewards * tf.cast(in_range, self.tf_float)

  @tf.function
  def polar_decomp(self, replay_rewards, head_id):
    """Decomposes a reward following Equation 13 of Fatemi & Tavakoli (2022).
    
    NOTES:
    - We assume rewards are not clipped by the environment and, instead, are 
    clipped internally here.

    - The functionality of this method is identical to using `config_1_decomp` 
    with the number of reward channels (`num_reward_channels`) set to two.
    """
    assert self.num_reward_channels == 2
    if head_id == 0:
      assert self.lambdas_on_heads[0] == 1 
      replay_rewards = self.rewards_above_a(replay_rewards, a=self.zero)
    elif head_id == 1:
      assert self.lambdas_on_heads[1] == -1
      replay_rewards = -1 * self.rewards_below_a(replay_rewards, a=self.zero)
    else:
      raise AssertionError
    return tf.minimum(self.one, replay_rewards)

  @tf.function
  def config_1_decomp(self, raw_replay_rewards, head_id):
    """Decomposes a reward following Configuration 2 of Fatemi & Tavakoli (2022)."""
    assert self.num_reward_channels / 2 in [1,2,3,4]
    if head_id < (self.num_reward_channels / 2):
      assert self.lambdas_on_heads[head_id] > 0
      replay_rewards = raw_replay_rewards 
      hid = head_id
    else:
      assert self.lambdas_on_heads[head_id] < 0
      assert head_id >= (self.num_reward_channels / 2) and head_id < self.num_reward_channels
      replay_rewards = -1 * self.rewards_below_a(raw_replay_rewards, a=self.zero)
      hid = head_id - self.num_reward_channels / 2
    
    if hid == 0:
      lambda1 = self.np_float(1)
      assert lambda1 == np.absolute(self.lambdas_on_heads[head_id])
      if self.num_reward_channels == 2:
        replay_rewards = self.rewards_above_a(replay_rewards, a=self.zero)
        return (self.np_float(1) / lambda1) * tf.minimum(self.one, replay_rewards)
      else:
        return (self.np_float(1) / lambda1) * self.rewards_between_a_and_b(replay_rewards, 
            a=self.zero, b=self.one, equal_a=True, equal_b=True)
    elif hid == 1:
      lambda2 = self.np_float(10)
      assert lambda2 == np.absolute(self.lambdas_on_heads[head_id])
      if self.num_reward_channels == 4:
        replay_rewards = self.rewards_above_a(replay_rewards, a=self.one)
        return (self.np_float(1) / lambda2) * tf.minimum(self.ten, replay_rewards)
      else:
        return (self.np_float(1) / lambda2) * self.rewards_between_a_and_b(replay_rewards, 
            a=self.one, b=self.ten, equal_a=False, equal_b=True)
    elif hid == 2:
      lambda3 = self.np_float(100)
      assert lambda3 == np.absolute(self.lambdas_on_heads[head_id])
      if self.num_reward_channels == 6:
        replay_rewards = self.rewards_above_a(replay_rewards, a=self.ten)
        return (self.np_float(1) / lambda3) * tf.minimum(self.hundred, replay_rewards)
      else:
        return (self.np_float(1) / lambda3) * self.rewards_between_a_and_b(replay_rewards, 
            a=self.ten, b=self.hundred, equal_a=False, equal_b=True)
    elif hid == 3:
      lambda4 = self.np_float(1000)
      assert lambda4 == np.absolute(self.lambdas_on_heads[head_id])
      if self.num_reward_channels == 8:
        replay_rewards = self.rewards_above_a(replay_rewards, a=self.hundred)
        return (self.np_float(1) / lambda4) * tf.minimum(self.thousand, replay_rewards)
      else:
        raise NotImplementedError
    else:
      raise NotImplementedError 

  @tf.function
  def config_2_decomp(self, raw_replay_rewards, head_id):
    """Decomposes a reward following Configuration 2 of Fatemi & Tavakoli (2022)."""
    assert self.num_reward_channels / 2 in [1,2,3,4]
    if head_id < (self.num_reward_channels / 2):
      assert self.lambdas_on_heads[head_id] > 0
      replay_rewards = self.rewards_above_a(raw_replay_rewards, a=self.zero)
      hid = head_id
    else:
      assert self.lambdas_on_heads[head_id] < 0
      assert head_id >= (self.num_reward_channels / 2) and head_id < self.num_reward_channels
      replay_rewards = -1 * self.rewards_below_a(raw_replay_rewards, a=self.zero)
      hid = head_id - self.num_reward_channels / 2
    
    if hid == 0:
      lambda1 = self.np_float(1)
      assert lambda1 == np.absolute(self.lambdas_on_heads[head_id])
      return (self.np_float(1) / lambda1) * tf.minimum(self.one, replay_rewards)
    elif hid == 1:
      lambda2 = self.np_float(9)
      assert lambda2 == np.absolute(self.lambdas_on_heads[head_id])
      replay_rewards = self.rewards_above_a(replay_rewards, a=self.one)
      replay_rewards = tf.math.subtract(replay_rewards, self.one)
      return (self.np_float(1) / lambda2) * tf.minimum(self.nine, replay_rewards)
    elif hid == 2:
      lambda3 = self.np_float(90)
      assert lambda3 == np.absolute(self.lambdas_on_heads[head_id])
      replay_rewards = self.rewards_above_a(replay_rewards, a=self.ten)
      replay_rewards = tf.math.subtract(replay_rewards, self.ten)
      return (self.np_float(1) / lambda3) * tf.minimum(self.ninety, replay_rewards)
    elif hid == 3:
      lambda4 = self.np_float(900)
      assert lambda4 == np.absolute(self.lambdas_on_heads[head_id])
      replay_rewards = self.rewards_above_a(replay_rewards, a=self.hundred)
      replay_rewards = tf.math.subtract(replay_rewards, self.hundred)
      return (self.np_float(1) / lambda4) * tf.minimum(self.ninehundred, replay_rewards)
    else:
      raise NotImplementedError

  @tf.function
  def two_ensemble_polar_decomp(self, replay_rewards, head_id): 
    """Polar reward decomposition for an ensemble of two learners."""
    assert self.num_reward_channels == 4
    if head_id == 0:
      assert self.lambdas_on_heads[0] == 0.5
      lambda1 = self.np_float(self.lambdas_on_heads[0]) 
      replay_rewards = lambda1 * self.rewards_above_a(replay_rewards, a=self.zero)
    elif head_id == 1:
      assert self.lambdas_on_heads[1] == -0.5
      lambda2 = self.np_float(self.lambdas_on_heads[1])
      replay_rewards = lambda2 * self.rewards_below_a(replay_rewards, a=self.zero)
    elif head_id == 2:
      assert self.lambdas_on_heads[2] == 0.5
      lambda3 = self.np_float(self.lambdas_on_heads[2])
      replay_rewards = lambda3 * self.rewards_above_a(replay_rewards, a=self.zero)
    elif head_id == 3:
      assert self.lambdas_on_heads[3] == -0.5
      lambda4 = self.np_float(self.lambdas_on_heads[3])
      replay_rewards = lambda4 * self.rewards_below_a(replay_rewards, a=self.zero)
    else:
      raise NotImplementedError
    return tf.minimum(self.half, replay_rewards)