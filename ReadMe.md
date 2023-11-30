# CartPole Q-Learning Project

This project is focused on implementing a Q-Learning algorithm to solve the CartPole problem from OpenAI's Gym. The primary objective of the game is to balance a pole on a cart for as long as possible by moving the cart left or right.

## Project Overview
This was my first reinforcement learning project, I had previously researched it a lot but never put 'pen to paper' and attempted a project! The CartPole environment and the Q-Learning algorithm seemed to be the obvious choice as the CartPole environment is a classic testbed for reinforcement learning algorithms. The Q-Learning approach used a discretised state space to train an agent to balance the pole. The agent learns to map states to actions in a way that maximizes the cumulative reward, aiming to keep the pole balanced for as long as possible.

## Features

- Implementation of the Q-Learning algorithm.
- Discretisation of continuous state space to a discrete state space
- Comparison of agent performance with a baseline random agent.
- Adaptive learning rate and exploration decay.

## Reward Metrics

- **Reward Structure:** The agent gets a reward of +1 every time the pole remains upright. The aim is for the pole to remain standing for as long as possible so the rewards accumulate with better performance.
- **Episode Termination:** The learning episode terminates, resetting the environment if:
  - The pole angle is more than ±12 degrees from vertical.
  - The cart moves more than 2.4 units from the center.
  - The maximum steps per episode are reached (set at 200 in this project).
- **Q-Table:** Stores the estimated rewards for state-action pairs and gets updated after each action. The algorithm considers the tradeoff between immediate and future rewards with the discount factor.

The total episode rewards were used as the evaluation metric, as the agent learns how to get more rewards the total episode rewards increases to the cut off limit (200).

## Versions

- **Version 1 (cartpole_Qlearning_old.py):** A rough version where I was getting to grips with the environment, the problem, and the algorithm. Extensive tuning still could only achieve an average improvement of ~+6 episode rewards over the baseline score of 21 rewards per episode - not very good!
- **Version 2 (cartpole_Qlearning.py):** Utilised the scikit-learn KBinsDiscretizer module to split the continuous state space into discrete bins. This allowed the agent to learn far more successfully, regularly meeting the cutoff of 200 rewards per episode after 5000 episodes and achieving an average of +100 over the baseline.


## Requirements

- Python 3.10
- see requirements.txt

![2500](https://github.com/trow-land/Reinforcement_Learning/blob/main/videos/2500_episodes.mp4)
