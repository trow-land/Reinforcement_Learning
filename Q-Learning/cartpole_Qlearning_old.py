import gym
import numpy as np
import random
import pygame

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')
#print(env.action_space.n)  # print the number of actions available (2)


# Defining the Q-table
# There are 2 actions available (move left & move right)
# There is a continuous number of states so that will have to be discretised into a discrete number of values

actions = env.action_space.n


Observation = [30, 30, 50, 50]  # # cart position, cart velocity, pole position, pole velocity
np_array_win_size = np.array([0.1, 0.1, 0.03, 0.05])  # The bin width
# Average reward of random agent: 21.8977
# Average reward of Q-learning agent: 27.6393
# Improvement over random: 5.741599999999998

# Observation = [20, 20, 40, 40]
# np_array_win_size = np.array([0.24, 0.5, 0.02, 0.1])
# Average reward of random agent: 22.317
# Average reward of Q-learning agent: 27.281
# Improvement over random: 4.963999999999999

# Observation = [10, 10, 20, 20]
# np_array_win_size = np.array([0.48, 1, 0.04, 0.2])
# Average reward of random agent: 22.2139
# Average reward of Q-learning agent: 23.6267
# Improvement over random: 1.4128000000000007

# Observation = [40, 40, 80, 80]
# np_array_win_size = np.array([0.12, 0.25, 0.01, 0.05])
# Average reward of random agent: 22.1223
# Average reward of Q-learning agent: 23.2471
# Improvement over random: 1.1248000000000005

# Observation = [15, 15, 30, 30]
# np_array_win_size = np.array([0.32, 0.8, 0.028, 0.16])
# Average reward of random agent: 22.2559
# Average reward of Q-learning agent: 14.8556
# Improvement over random: -7.4003

# Observation = [25, 25, 50, 50]
# np_array_win_size = np.array([0.192, 0.4, 0.0168, 0.12])
# Average reward of random agent: 22.1474
# Average reward of Q-learning agent: 24.6541
# Improvement over random: 2.5066999999999986

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))  # random between 0 and 1
#q_table = np.zeros(Observation + [env.action_space.n])  # zeros
q_table.shape
total_rewards = []

initial_state = env.reset()
print(initial_state)  # (array([ 0.00858198, -0.02165736, -0.01722066, -0.00183534], dtype=float32), {})

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

def descritise_state(state_tuple):
    state = state_tuple[0]  # current continuous value of the environment's state
    shift_values = np.array([15, 10, 1, 10])  # shift values for discretization
    discrete_state = state / np_array_win_size + shift_values  # result of discretisation offset to keep pos. Result is a scaled and shifted version of continuous values
    
    # Apply clamping to each element of the discrete state
    discrete_state_int = [clamp(int(discrete_state[i]), 0, Observation[i] - 1) for i in range(len(discrete_state))]

    return tuple(discrete_state_int)


# test the discretise_state function with the initial state
discretised_state = descritise_state(initial_state)


learning_rate = 0.1
discount_factor = 0.95
exploration_factor = 1
decay_factor = 0.9995


#################### Implement the learning loop ####################

num_episodes = 10000
max_steps_per_episode = 200



for episode in range(num_episodes):
    state = env.reset()
    #print("Continuous state: ", state)  # example output -> Continuous state:  (array([ 0.01919788, -0.02791256, -0.0057743 , -0.041681  ], dtype=float32), {})
    current_state = descritise_state(state)

    total_reward = 0

    for step in range(max_steps_per_episode):
        # if step % 100 == 0:
        #     env.render()

        number = random.uniform(0,1)
        if number >= exploration_factor:  # When number is above exploration factor then exploit
            action = np.argmax(q_table[current_state])  # choose the highest value in the q table
        else:  # explore 
            
            action = env.action_space.sample()  # choose a random action

        total_reward += 1  # reward for every step the pole remains up
        

        # take the action
        new_state, reward, done, extra_boolean, info = env.step(action)
        new_state = descritise_state(new_state)  # New_state:  [-0.00953921  0.15152529  0.04431454 -0.24312742]
        #print("New_state_discrete: ", new_state)  # New_state_discrete:  (14, 9, 0, 9)

        # update the q value
        old_value = q_table[current_state][action]
        next_max = np.max(q_table[new_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max) # use the q-learning equation to update 

        q_table[current_state][action] = new_value
        current_state = new_state

        # state_tuple = tuple(int(s) for s in state)
        # action_int = int(action)
        # new_element = state_tuple + (action_int,)

        if done:
            break

    
    # Decrease exploration rate over time but hold steady for x episodes
    if episode > 1000 and exploration_factor > 0.01:  
        if total_reward > prior_reward:  # if the model has performed better than the previous episode
            exploration_factor *= decay_factor  # exploration decay

    # Print episode stats
    # if episode % 500 == 0:
    #     print(f"Episode {episode}, Exploration rate: {exploration_factor:.2f}, Episode rewards: {total_reward}")
    
    total_rewards.append(total_reward)
    prior_reward = total_reward


def baseline_agent(env, num_episodes, max_steps_per_episode):
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


random_rewards = baseline_agent(env, 10000, 200) 
average_random_reward = sum(random_rewards) / len(random_rewards)
print(f"Average reward of random agent: {average_random_reward}")


average_q_learning_reward = sum(total_rewards) / len(total_rewards)
print(f"Average reward of Q-learning agent: {average_q_learning_reward}")

# Compare
print(f"Improvement over random: {average_q_learning_reward - average_random_reward}")




