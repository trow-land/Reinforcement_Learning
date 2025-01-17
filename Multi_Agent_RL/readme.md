# Multi-Agent Predator-Prey Simulation with Q-Learning

## Overview

This project simulates a **multi-agent predator-prey** environment where a fox (predator) attempts to catch multiple rabbits (prey) in a grid world. The fox and rabbits use **Q-Learning** to learn their optimal movement strategies over time.

The simulation uses a custom-built GridWorld environment where:
- The **fox (predator)** learns to pursue and catch the rabbits by reducing the distance between them.
- The **rabbits (prey)** attempt to avoid the fox by moving away from it.

Both agents (fox and rabbits) update their strategies over episodes using reinforcement learning.

## Features

- **GridWorld Environment:** A customisable grid that visualises the fox and rabbit positions.
- **Q-Learning Agent:** Both the predator (fox) and prey (rabbits) use Q-Learning to optimise their actions based on rewards.
- **Dynamic Visualisation:** The grid updates in real-time, displaying the movement of the fox and rabbits across the environment.
- **Multiple Preys:** Several rabbits can be initialised and will move independently of each other to avoid being caught.
- **Performance Tracking:** Track the steps taken by the fox to catch each rabbit and plot the learning progress over episodes.

## Project Structure

- `GridWorld`: A class representing the environment where the predator and prey agents interact.
- `QLearningAgent`: A class implementing the Q-Learning algorithm for the fox and rabbit agents.
- `qlearning_loop`: The main loop where episodes run, and agents interact and learn from their environment.

## Setup

### Prerequisites

This project requires the following dependencies:
- `Python 3.7+`
- `numpy`
- `matplotlib`
- `tqdm`

### Parameters

- **Grid Size:** You can modify the grid size by passing the desired size during the GridWorld initialisation.
- **Number of Rabbits:** The number of prey agents (rabbits) can be customised by adjusting the `num_rabbits` parameter in the simulation loop.
- **Q-Learning Hyperparameters:** The learning rate, discount factor, and epsilon decay can be modified in the `QLearningAgent` class initialisation.

## Visualisation

The simulation includes a dynamic grid visualisation that updates as the predator and prey agents move. After the learning loop, the project also provides a performance plot, showing how the fox's strategy improves over time.

### Example Visualisation

- **Fox (Predator):** Represented by the number `1` on the grid.
- **Rabbits (Prey):** Represented by the number `2` on the grid.
- **Grid:** Visualises the positions of the fox and rabbits over the course of the simulation.

![gridworld](https://github.com/trow-land/Reinforcement_Learning/blob/main/Multi_Agent_RL/gridworld.png)

## Q-Learning Implementation

The agents learn their strategies through **Q-Learning**, a reinforcement learning algorithm. The fox is rewarded for moving closer to the rabbits and penalised for moving farther away. Similarly, the rabbits are rewarded for increasing their distance from the fox and penalised when they are caught.

### Key Formulas:

#### Q-Value Update:

**Q(s, a) ← Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))**

Where:
- `s`: Current state
- `a`: Action taken
- `r`: Reward received
- `s'`: New state after action `a`
- `α`: Learning rate
- `γ`: Discount factor

With a single Fox and Rabbit the fox quickly improves towards and optimal policy

![training_graph](https://github.com/trow-land/Reinforcement_Learning/blob/main/Multi_Agent_RL/output.png)

## Customisation

You can customise the following parameters:
- **Grid size**: Modify the grid size by changing the `size` parameter in the `GridWorld` class.
- **Number of rabbits**: Change the number of rabbits in the environment by setting the `num_rabbits` parameter in the `reset_env` function.
- **Q-Learning Hyperparameters**: Adjust the learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`) in the `QLearningAgent` class.

## Fox and Rabbit Learning
The purpose of this project has been to determine how the cunning fox and hunt x number of rabbits in a simple simulation. Using a single fox with multiple prey has given a few opportunities to disadvantage the rabbits.

1) Initialising the rabbits with a slower learning rate.
2) Gives the rabbits few learning opportunities
2) rabbits prefer exploration vs exploiting their knowledge

The below plot demonstrates how the fox learns to effectively hunt the multiple rabbits throughout the training episodes

![multirabbit](https://github.com/trow-land/Reinforcement_Learning/blob/main/Multi_Agent_RL/multi_rabbit.png)

## Future Enhancements

- **Shared Learning**: Rabbits should be able to share knowledge between training episoed
- **More complex environments**: Introduce obstacles in the grid that both the fox and rabbits have to navigate around.
- **Advanced learning algorithms**: Implement more advanced algorithms like Deep Q-Learning (DQN) to improve learning performance.
- **Multi-agent interactions**: Explore scenarios with multiple foxes (predators), limiting the field of view to encourage cooperation

## License

This project is licensed under the MIT License.

