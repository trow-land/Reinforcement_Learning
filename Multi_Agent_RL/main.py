import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


class GridWorld:

    def __init__(self, size=8) -> None:
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # initialised grids where 0 is an empty grid
        self.predators = []
        self.prey = []

    def add_agents(self, position, agent_type):
        if agent_type == 'predator':
            self.predators.append(position)
            self.grid[position[0], position[1]] = 1  # 1 represents the alpha (predator)

        elif agent_type == 'prey':
            self.prey.append(position)
            self.grid[position[0], position[1]] = 6  # 2 represents the prey

    def reset_env(self):
        self.predators = []
        self.prey = []
        self.grid = np.zeros((self.size, self.size), dtype=int)


    def print_grid(self):
        print(self.grid)


    def agent_step(self, agent_position, action):
        x, y = agent_position

        if action == 'up':
            y += 1
            y = min(y, self.size - 1)
        elif action == 'down':
            y -= 1
            y = max(y, 0)
        elif action == 'left':
            x -= 1
            x = max(x, 0)
        elif action == 'right':
            x += 1
            x = min(x, self.size - 1)

        if self.grid[x, y] != self.grid[agent_position[0], agent_position[1]]:
            # make the move
            self.grid[x, y] = self.grid[agent_position[0], agent_position[1]]
            # reset original position to background colour
            self.grid[agent_position] = 0
        else:
            # agent is at a boundary and hasnt moved
            pass

        return (x, y)


    def visualize_grid_dynamic(self):
        plt.clf() # clear plot
        plt.imshow(self.grid, cmap='Pastel1', origin='upper')
        plt.title('GridWorld')
        plt.grid(True)
        plt.draw()  # new plot
        plt.pause(0.5) 

# moving matplotlib
plt.ion()

# Create environment
env = GridWorld(size=8)
env.add_agents((1, 1), 'predator')
env.add_agents((6, 7), 'prey')

possible_actions = ['up', 'down', 'left', 'right']

# Simulate random movement until prey is caught
while env.prey[0] != env.predators[0]:
    prey_move = possible_actions[np.random.randint(0, 4)]  # Random prey move
    predator_move = possible_actions[np.random.randint(0, 4)]  # Random predator move

    # Move prey and predator
    env.prey[0] = env.agent_step(env.prey[0], prey_move)
    env.predators[0] = env.agent_step(env.predators[0], predator_move)

    env.visualize_grid_dynamic()

    #time.sleep(0.5)

plt.ioff()
plt.show()
    
