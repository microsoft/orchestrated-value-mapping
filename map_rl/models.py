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
"""Atari-specific utilities including Atari-specific network architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

import gin


MapDQNNetworkType = collections.namedtuple('map_dqn_network', 
    ['q_values', 'q_values_on_heads', 'q_tilde_values_on_heads'])



@gin.configurable
class MapDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's mapped Q-values."""

  def __init__(self, num_actions, lambdas_on_heads, inverse_map_funcs, 
               map_func_params, initializers_on_heads, use_gradscaling, 
               use_nonlinear_heads, tf_float, name=None):
    super(MapDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    self.lambdas_on_heads = lambdas_on_heads
    self.inverse_map_funcs = inverse_map_funcs
    self.map_func_params = map_func_params
    self.use_gradscaling = use_gradscaling
    self.use_nonlinear_heads = use_nonlinear_heads
    self.tf_float = tf_float
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    if not use_nonlinear_heads:
      self.dense_torso = tf.keras.layers.Dense(512, activation=activation_fn,
                                               name='fully_connected')
    else: 
      self.dense_heads_torso = []

    self.dense_heads = []
    for kernel_init, bias_init in zip(initializers_on_heads['kernel_initializers'], 
                                      initializers_on_heads['bias_initializers']):
      if use_nonlinear_heads:
        self.dense_heads_torso.append(
            tf.keras.layers.Dense(512, activation=activation_fn, 
                                  name='fully_connected'))
      if type(bias_init) == float:
        bias_init = tf.keras.initializers.Constant(bias_init)
      else:
        assert bias_init == 'zeros'
      self.dense_heads.append(
          tf.keras.layers.Dense(num_actions, kernel_initializer=kernel_init, 
                                bias_initializer=bias_init, name='fully_connected'))

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    if not self.use_nonlinear_heads:
      x = self.dense_torso(x)
    if self.use_gradscaling:
      N = len(self.lambdas_on_heads)
      gradscale_mult = 1.0 / np.sqrt(N)
      t = tf.multiply(x, gradscale_mult)
      x = t + tf.stop_gradient(x - t)
    x = tf.cast(x, self.tf_float)

    q_tilde_values_on_heads = []
    q_values_on_heads = []
    # Form of 'map_func_params', e.g., for the polar reward decomposition: 
    # [[pos_Delta, c], [neg_Delta, c]]. Each internal list is for one head.
    for hid, (head_dense, head_map_params, head_inverse_map_func) in enumerate(zip(
        self.dense_heads, self.map_func_params, self.inverse_map_funcs)):
      if self.use_nonlinear_heads:
        q_tilde_values = self.dense_heads_torso[hid](x)
        q_tilde_values = head_dense(q_tilde_values)
      else:
        q_tilde_values = head_dense(x)
      q_tilde_values_on_heads.append(q_tilde_values)
      # Inverse mapping.
      q_values_on_heads.append(head_inverse_map_func(q_tilde_values, *head_map_params))
  
    # Aggregate Q-values across heads. 
    first_head = True
    for head_lambda, head_q_values in zip(self.lambdas_on_heads, q_values_on_heads):
      if first_head:
        q_values = head_lambda * head_q_values
        first_head = False
      else:
        q_values += head_lambda * head_q_values

    return MapDQNNetworkType(q_values, q_values_on_heads, q_tilde_values_on_heads)