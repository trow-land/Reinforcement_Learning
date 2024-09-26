import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import math


class GridWorld:

    def __init__(self, size=8) -> None:
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # initialize grid
        self.predator = []
        self.prey = []

        # Define the custom colormap
        self.cmap = colors.ListedColormap(['forestgreen', 'chocolate', 'slategrey'])  # Index 0: green, 1: red, 2: grey
        self.bounds = [0, 1, 2, 3]  # Define boundaries for colormap
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

    def add_agents(self, position, agent_type):
        if agent_type == 'predator':
            self.predator = position  # Store predator position
            self.grid[position[0], position[1]] = 1  # 1 represents the predator

        elif agent_type == 'prey':
            self.prey = position  # Store prey position
            self.grid[position[0], position[1]] = 2  # 2 represents the prey

    def reset_env(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.predator = (1, 1)
        self.prey = (6, 7)
        self.grid[self.predator[0], self.predator[1]] = 1
        self.grid[self.prey[0], self.prey[1]] = 2
        return self.predator, self.prey

    def agent_step(self, agent_position, action):
        x, y = agent_position
        if action == 'up':
            y = min(y + 1, self.size - 1)
        elif action == 'down':
            y = max(y - 1, 0)
        elif action == 'left':
            x = max(x - 1, 0)
        elif action == 'right':
            x = min(x + 1, self.size - 1)

        return (x, y)

    def move_prey_randomly(self, prey_state):
        possible_actions = ['up', 'down', 'left', 'right']
        action = np.random.choice(possible_actions)
        prey_position = self.agent_step(prey_state, action)
        return prey_position

    def visualise_grid_dynamic(self, episode):
        plt.clf()
        plt.imshow(self.grid, cmap=self.cmap, norm=self.norm, origin='upper')
        plt.title(f'Fox and Rabbit : Episode {episode}')
        plt.axis('off')  # Remove axes for a cleaner look
        plt.draw()
        plt.pause(0.1)


class QLearningAgent():

    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=1) -> None:
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # define qtable
        self.qtable = np.zeros((grid_size, grid_size, grid_size, grid_size, 4))  # 2d grid and 4 actions 
        self.actions = ['up', 'down', 'left', 'right']


    def choose_action(self, predator_state, prey_state):
        state = (predator_state[0], predator_state[1], prey_state[0], prey_state[1])

        # exploration
        random_number = np.random.uniform(0, 1)
        # print(random_number)
        if random_number < self.epsilon:
            return np.random.randint(0, 4)  # Return a random action index (0 to 3)
        else:
            # exploitation
            #print(self.qtable[predator_state[0], predator_state[1], prey_state[0], prey_state[1]])
            return np.argmax(self.qtable[state[0], state[1], state[2], state[3]])  # Return the best action index


    def update_qtable(self, predator_state, action, reward, next_predator_state, prey_state):
        state = (predator_state[0], predator_state[1], prey_state[0], prey_state[1])
        next_state = (next_predator_state[0], next_predator_state[1], prey_state[0], prey_state[1])

        # best future action
        best_move = np.argmax(self.qtable[next_state[0], next_state[1], next_state[2], next_state[3]])

        # Qlearning formula
        self.qtable[state[0], state[1], state[2], state[3], action] += self.alpha * (
            reward + self.gamma * self.qtable[next_state[0], next_state[1], next_state[2], next_state[3], best_move] -
            self.qtable[state[0], state[1], state[2], state[3], action]
        )

    def agent_step(self, agent_position, action_index):
        action = self.actions[action_index]  # Map index to action string
        x, y = agent_position

        if action == 'up':
            y += 1
            y = min(y, self.grid_size - 1)
        elif action == 'down':
            y -= 1
            y = max(y, 0)
        elif action == 'left':
            x -= 1
            x = max(x, 0)
        elif action == 'right':
            x += 1
            x = min(x, self.grid_size - 1)

        return (x, y)


    def get_reward(self, predator_pos, prey_pos, prev_distance):
        distance = math.dist(predator_pos, prey_pos)
        if distance == 0:
            reward = 10  # High reward for catching the prey
        elif distance < prev_distance:
            reward = 1  # Small reward for getting closer
        else:
            reward = -1  # Penalty for moving farther away
        return reward, distance


# moving matplotlib
plt.ion()

g_size = 8

# Create environment
env = GridWorld(size=g_size)
env.add_agents((1, 1), 'predator')
env.add_agents((6, 7), 'prey')

agent = QLearningAgent(grid_size=g_size)

pred_caught = []

def qlearning_loop(episodes, max_steps):
    for episode in range(episodes):
        predator_state, prey_state = env.reset_env()
        agent.epsilon = max(0.1, agent.epsilon * 0.995)
        prev_distance = math.dist(predator_state, prey_state)  # Track initial distance

        for step in range(max_steps):
            # move rabbit
            env.grid[env.prey[0], env.prey[1]] = 0
            prey_state = env.move_prey_randomly([env.prey[0], env.prey[1]])
            env.prey = prey_state
            env.grid[env.prey[0], env.prey[1]] = 6

            # fox chooses its action based upon the state of its position and the rabbits position
            action = agent.choose_action(predator_state, prey_state)

            # move fox
            env.grid[predator_state[0], predator_state[1]] = 0
            new_predator_state = agent.agent_step(predator_state, action)
            env.grid[new_predator_state[0], new_predator_state[1]] = 1 

            # calc reward and update the qtable
            reward, prev_distance = agent.get_reward(new_predator_state, prey_state, prev_distance)
            agent.update_qtable(predator_state, action, reward, new_predator_state, prey_state)

            predator_state = new_predator_state  # Update predator state

            # if episode % 1000 == 0:
            # # Visualise the grid every x steps
            #     env.visualise_grid_dynamic(episode)

            if new_predator_state == prey_state:
                pred_caught.append((episode, step + 1))  # Store steps for plotting the learning of the agent
                break
            elif step == max_steps - 1:
                pred_caught.append((episode, step + 1))
            
            



# Start learning loop
plt.ioff()

qlearning_loop(episodes=2500, max_steps=100)

# extract episode numbers and steps
episodes = [x[0] for x in pred_caught]
steps_to_catch = [x[1] for x in pred_caught]

window_size = 50
moving_avg = [np.mean(steps_to_catch[max(0, i - window_size):(i + 1)]) for i in range(len(steps_to_catch))]
plt.plot(episodes, moving_avg)
plt.xlabel(xlabel="Learning Episodes")
plt.ylabel(ylabel="Fox steps to catch rabbit")
plt.title("Fox Learning")
plt.show()
