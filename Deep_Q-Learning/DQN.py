from sklearn.preprocessing import KBinsDiscretizer
import gym
import numpy as np
import random
import pygame
import time, math, random
import tensorflow as tf
import collections

from tf_agents.agents.dqn import dqn_agent
from typing import Tuple


env = gym.make("CartPole-v1", render_mode='human')
print(env.action_space.n)  # 2
print(env.observation_space)  # Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
print(env.reward_range)  # (-inf, inf)
env.reset()

## DQNs do not need to discretise the state-space like normal Q-learning

state = env.reset()
print(state)  # (array([0.03922939, 0.02864914, 0.01625022, 0.02568655], dtype=float32), {})

# the state is a tuple containing the state space array and a blank dictionary

state_space = state[0]  # seperating the state information (4 item tuple)

cartPosition, cartVelocity, poleAngle, poleVelocity = state_space  # initial input values


# Define network
def create_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, (input_shape=env.observation_space[0],), activation='relu', name='input_layer'),
        tf.keras.layers.Dense(8, activation='relu', name='hidden1'),
        tf.keras.layers.Dense(24, activation='relu', name='hidden2'),
        tf.keras.layers.Dense(2, activation='linear', name='output')
        ])
    return model

class ReplayBuffer:
    def __init__(self, capacity):
        # capacity: max number of experiences held
        # older experiences discarded once capacity is reached
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # adds new experiences to the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        minibatch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)
    
# Initialise buffer
buffer_size = 10000
replay_buffer = ReplayBuffer(buffer_size)


# Initial hyperparams before any tuning
discount_factor = 0.95
exploration_factor = 0.9
decay_factor = 0.9995


class DQN_Agent:

    def __init__(self, learning_rate, discount_factor, exploration_factor, decay_factor, batch_size):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor * decay_factor
        self.decay_factor = decay_factor
        self.batch_size = batch_size
        
        # create both networks
        self.q_network = create_network()
        self.q_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        self.target_network = create_network()
        self.target_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        self.target_network.set_weights(self.q_network.get_weights())


    def select_action(self, state):

        number = np.random(0,1)
        if number >= self.exploration_factor:
            # exploit
            q_values = self.q_network.predict(np.array([state]))
            action = np.argmax(q_values)
        else:
            # explore
            action = env.action_space.sample()
        
        self.exploration_factor *= self.decay_factor

        return action












