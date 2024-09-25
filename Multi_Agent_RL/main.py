import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


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


    def visualise_grid(self):
        plt.imshow(self.grid, cmap='Pastel1', origin='upper')
        plt.title('GridWorld')
        plt.grid(False)
        plt.show()

    def agent_step(self, agent_position, action):
        x, y = agent_position

        if action == 'up':
            y -= 1
            y = max(0, y)
        elif action == 'down':
            y += 1
            y = min(y, self.size - 1)
        elif action == 'left':
            x -= 1
            x = max(0, x)
        elif action == 'right':
            x += 1
            x = min(x, self.size - 1)

        # make the move
        self.grid[x, y] = self.grid[agent_position]

        # reset original position to background colour
        self.grid[agent_position] = 0
        
        
        




# create a gridworld
env = GridWorld()

# add some agents
env.add_agents((0,2), 'prey')
env.add_agents((6,7), 'predator')

env.visualise_grid()

env.agent_step(env.prey[0], 'down')

env.visualise_grid()

env.agent_step(env.predators[0], 'up')
env.visualise_grid()


# define action space
possible_actions = ['up', 'down', 'left', 'right']

# implement some random actions for both predator and prey and watch them navigate the environment
while env.prey[0] != env.predators[0]:
    pass
