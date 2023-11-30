### Trying to implement a new discretisation function ---- not working yet
from sklearn.preprocessing import KBinsDiscretizer
import gym
import numpy as np
import random
import pygame
import time, math, random
from typing import Tuple


env = gym.make('CartPole-v1', render_mode='human')
#env = gym.make('CartPole-v1')  # for quicker training
#print(env.action_space.n)  # print the number of actions available (2)


#actions = env.action_space.n


n_bins = ( 6 , 12 )  # 6 bins for pole and 12 for cart
lower_bounds = [ env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[2], math.radians(50) ]

def discretise_state( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continuous state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))


q_table = np.zeros(n_bins + (env.action_space.n,))  # zeros
q_table.shape
total_rewards = []


discount_factor = 0.95
exploration_factor = 1
decay_factor = 0.9995

# Adaptive learning of Learning Rate
def learning_rate(n : int , min_rate=0.01 ) -> float  :
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))




#################### Implement the learning loop ####################

num_episodes = 10000
max_steps_per_episode = 150

for episode in range(num_episodes):
    state = env.reset()
    current_state = discretise_state(*state[0])
    total_reward = 0

    if episode % 500 == 0 and episode >= (num_episodes / 4) :
        env.render()


    for step in range(max_steps_per_episode):
        

        number = random.uniform(0,1)
        if number >= exploration_factor:  # When number is above exploration factor then exploit
            action = np.argmax(q_table[current_state])  # exploit - choose the highest value in the q table
        else: 
            action = env.action_space.sample()  # explore - choose a random action

        total_reward += 1  # reward for every step the pole remains up

        # take the action
        new_state, reward, done, extra_boolean, info = env.step(action)
        new_state = discretise_state(*new_state)

        
        next_max = np.max(q_table[new_state])  # maximum estimated futuree reward
        learned_value = reward + discount_factor + next_max
        old_value = q_table[current_state][action]  # IndexError: arrays used as indices must be of integer (or boolean) type
        new_value = (1 - learning_rate(episode)) * old_value + learning_rate(episode) * learned_value  # # use the q-learning equation to update
        q_table[current_state][action] = new_value
        current_state = new_state

        if done:
            break

    
    # Decrease exploration rate over time but hold steady for x episodes
    if episode > 1000 and exploration_factor > 0.01:  
        if total_reward > prior_reward:  # if the model has performed better than the previous episode
            exploration_factor *= decay_factor  # exploration decay

    # Print episode stats
    if episode % 500 == 0:
        print(f"Episode {episode}, Exploration rate: {exploration_factor:.2f}, Episode rewards: {total_reward}")
    
    total_rewards.append(total_reward)
    prior_reward = total_reward

# Defining a baseline agent for comparison and proof of learning 
def random_agent(env, num_episodes, max_steps_per_episode):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = env.action_space.sample()  # Random action
            new_state, reward, done, extra_boolean, _ = env.step(action)
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

    return total_rewards


random_rewards = random_agent(env, 10000, 150) 
average_random_reward = sum(random_rewards) / len(random_rewards)
average_q_learning_reward = sum(total_rewards) / len(total_rewards)


print(f"Average reward of baseline agent: {average_random_reward}")
print(f"Average reward of Q-learning agent: {average_q_learning_reward}")
print(f"Improvement over random: {average_q_learning_reward - average_random_reward}")




