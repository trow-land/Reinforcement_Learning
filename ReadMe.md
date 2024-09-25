# CartPole Q-Learning Project

This project is focused on implementing a Q-Learning algorithm to solve the CartPole problem from OpenAI's Gym. The primary objective of the game is to balance a pole on a cart for as long as possible by moving the cart left or right.

## Project Overview
This was my first reinforcement learning project, I had previously researched it a lot but never put 'pen to paper' and attempted a project! The CartPole environment and the Q-Learning algorithm seemed to be the obvious choice as the CartPole environment is a classic testbed for reinforcement learning algorithms. The Q-Learning approach used a discretised state space to train an agent to balance the pole. The agent learns to map states to actions in a way that maximizes the cumulative reward, aiming to keep the pole balanced for as long as possible.

## Features

- Implementation of the Q-Learning algorithm.
- Discretization of continuous state space to a discrete state space
- Comparison of agent performance with a baseline random agent.
- Adaptive learning rate and exploration decay.

## Reward Metrics

- Reward Structure: The agent gets a reward of +1 everytime the pole remains upright. The aim is for the pole to remain standing for as long as possible so the rewards accumulate with better performance
- The learning episode terminates, resetting the environment if:
    - The pole angle +- 12 degrees from vertical
    - The cart moves 2.4 units from the center
    - The maximum steps are reached per episode (which I set at 200)
- The Q-table stores the estimated rewards for state-action pairs and it gets updated after each action. The algorithm considers the tradeoff between immediate and future reward with the discount factor. 

The total episode rewards were used as the evaluation metric, as the agent learns how to get more rewards the total episode rewards increases to the cut off limit (200).

## Versions

- Version 1 (cartpole_Qlearning_old.py) was a rough version where I was getting to grips with the environment, the problem and the algorithm. Extensive tuning still could only achieve an average improvement of ~+6 episode rewards over the baseline score of 21 rewards per episode - not very good!

- Version 2 ((cartpole_Qlearning.py)) used a the scikit-learn KBinsDiscretizer module to split the continous state space into discrete bins. This allowed the agent to learn far more successfully with the agent regularly meeting the cut-off of 200 rewards per episode after 5000 episodes and achieving an average of +100 over the baseline.
  
![training_log](https://github.com/trow-land/Machine-Learning/assets/75323342/f62f97a5-e23f-40d3-8df9-9a5b6bee10a9)

The above figure shows the agents improvement throughout traininng. The exploration decay which dictates the probability of the agent exploiting its knowledge rather than exploring was restricted in the early stage of training to allow the the agent to sufficiently explore and understand its environment.


  
  ![cartpole](https://github.com/trow-land/Machine-Learning/assets/75323342/d0198756-c023-4b41-ac16-a673f18c81f2)


