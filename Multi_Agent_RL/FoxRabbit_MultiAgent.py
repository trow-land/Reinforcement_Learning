import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import math
from tqdm import tqdm


class GridWorld:

    def __init__(self, size=8) -> None:
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # initialize grid
        self.predator = []
        self.preys = {}

        # Define the custom colormap
        self.cmap = colors.ListedColormap(['forestgreen', 'chocolate', 'slategrey'])  # Index 0: green, 1: red, 2: grey
        self.bounds = [0, 1, 2, 3]  # Define boundaries for colormap
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

    def add_agents(self, position, agent_type):
        if agent_type == 'predator':
            self.predator = position  # Store predator position
            self.grid[position[0], position[1]] = 1  # 1 represents the predator

        elif agent_type == 'prey':
            self.preys = position  # Store prey position
            self.grid[position[0], position[1]] = 2  # 2 represents the prey

    def reset_env(self, num_rabbits):
        self.grid = np.zeros((self.size, self.size), dtype=int)

        self.predator = (np.random.randint(0,g_size), np.random.randint(0,self.size))
        self.grid[self.predator[0], self.predator[1]] = 1

        # resetting the prey as a dictionary
        self.preys = {i : (np.random.randint(0,g_size), np.random.randint(0,self.size)) for i in range(num_rabbits)}

        for prey_pos in self.preys.values():
            self.grid[prey_pos[0], prey_pos[1]] = 2  # Set each rabbit position on the grid

        return self.predator, self.preys

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


    def choose_fox_action(self, predator_state, prey_states): 

        # identify the nearest rabbit to the fox
        distances = {rabbit_id: math.dist(predator_state, prey_state) for rabbit_id, prey_state in prey_states.items()}
        closest_rabbit_id = min(distances, key=distances.get)  # Find the rabbit ID with the smallest distance
        closest_rabbit = prey_states[closest_rabbit_id]  # Get the position of the closest rabbit

        state = (predator_state[0], predator_state[1], closest_rabbit[0], closest_rabbit[1])

        # exploration
        random_number = np.random.uniform(0, 1)
        # print(random_number)
        if random_number < self.epsilon:
            return np.random.randint(0, 4)  # Return a random action index (0 to 3)
        else:
            # exploitation
            #print(self.qtable[predator_state[0], predator_state[1], prey_state[0], prey_state[1]])
            return np.argmax(self.qtable[state[0], state[1], state[2], state[3]])  # Return the best action index
        

    def choose_rabbit_action(self, predator_state, prey_state):
        state = (predator_state[0], predator_state[1], prey_state[0], prey_state[1])
        # exploration
        random_number = np.random.uniform(0, 1)
        # print(random_number)
        if random_number < self.epsilon:
            return np.random.randint(0, 4)  # Return a random action index (0 to 3)
        else:
            # exploitation
            return np.argmax(self.qtable[state[0], state[1], state[2], state[3]])  # Return the best action index


    def update_fox_qtable(self, predator_state, action, reward, next_predator_state, prey_states):

        # identify the nearest rabbit to the fox
        distances = {rabbit_id: math.dist(predator_state, prey_state) for rabbit_id, prey_state in prey_states.items()}
        closest_rabbit_id = min(distances, key=distances.get)  # Find the rabbit ID with the smallest distance
        closest_rabbit = prey_states[closest_rabbit_id]

        state = (predator_state[0], predator_state[1], closest_rabbit[0], closest_rabbit[1])
        next_state = (next_predator_state[0], next_predator_state[1], closest_rabbit[0], closest_rabbit[1])

        # best future action
        best_move = np.argmax(self.qtable[next_state[0], next_state[1], next_state[2], next_state[3]])


        # Qlearning formula
        self.qtable[state[0], state[1], state[2], state[3], action] += self.alpha * (
            reward + self.gamma * self.qtable[next_state[0], next_state[1], next_state[2], next_state[3], best_move] -
            self.qtable[state[0], state[1], state[2], state[3], action]
        )

    def update_rabbit_qtable(self, predator_state, action, reward, next_predator_state, prey_state):


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


    def get_fox_reward(self, predator_pos, prey_positions, prev_distance):
        # The reward is set so one by one the fox goes for the nearest rabbit

        # Create a dictionary of distances with rabbit IDs as keys
        distances = {rabbit_id: math.dist(predator_pos, prey_state) for rabbit_id, prey_state in prey_positions.items()}

        # Find the rabbit with the minimum distance
        closest_rabbit_id = min(distances, key=distances.get)
        closest_rabbit = prey_positions[closest_rabbit_id]  # Get the position of the closest rabbit
        distance = distances[closest_rabbit_id]  # The actual distance to the closest rabbit

        # Determine reward based on distance
        if distance == 0:
            reward = 10  # High reward for catching the prey
        elif distance < prev_distance:
            reward = 1  # Small reward for getting closer
        else:
            reward = -1  # Penalty for moving farther away
        return reward, distance
    
    
    def get_rabbit_reward(self, predator_pos, prey_pos, prev_distance):
        distance = math.dist(predator_pos, prey_pos)
        if distance == 0:
            reward = -10  # High negative reward for getting caught
        elif distance < prev_distance:
            reward = -1  # Small penalty for getting closer
        else:
            reward = 1  # Penalty for moving farther away
        return reward, distance


def find_nearest_rabbit(predator_position, rabbit_positions):

    # identify the nearest rabbit to the fox
        distances = [math.dist(predator_position, prey_position) for prey_position in rabbit_positions.values()]
        closest_index = np.argmin(distances)
        closest_rabbit = rabbit_positions[closest_index]
        distance = math.dist(predator_position, closest_rabbit)

        return distance

# moving matplotlib
plt.ion()

g_size = 16

# Create environment
env = GridWorld(size=g_size)
env.add_agents((np.random.randint(0,g_size), np.random.randint(0,g_size)), 'predator')
env.add_agents((np.random.randint(0,g_size), np.random.randint(0,g_size)), 'prey')

fox = QLearningAgent(grid_size=g_size)  # start off with the same hyper params for pred or prey. Will reduce rabbits later because foxes are cunning


#rabbit = QLearningAgent(grid_size=g_size, alpha=0.05)

num_rabbits = 24
rabbits = [QLearningAgent(grid_size=g_size, alpha=0.05) for _ in range(num_rabbits)]
rabbits_caught_steps = {rabbit_id: [] for rabbit_id in range(num_rabbits)}

catch_steps = []  # steps taken to catch rabbit
rabbits_caught = 0  # number of rabbits caught

rabbits_left_record =[]

def qlearning_loop(episodes, max_steps, rabbits_caught):

    
    for episode in tqdm(range(episodes)):

        rabbits_left_at_episode = num_rabbits
        predator_state, prey_states = env.reset_env(num_rabbits)

        fox.epsilon = max(0.1, fox.epsilon * 0.995)

        for rabbit in rabbits:
            rabbit.epsilon = max(0.5, rabbit.epsilon * 0.995)
        
        prev_distance = find_nearest_rabbit(predator_state, prey_states)

        for step in range(max_steps):

            # Iterate through rabbits
            for rabbit_id, rabbit_position in list(env.preys.items()):

                rabbit = rabbits[rabbit_id]  # Get specific rabbit agent

                # Move rabbit
                env.grid[rabbit_position[0], rabbit_position[1]] = 0  # clear old position
                rabbit_action = rabbit.choose_rabbit_action(predator_state, rabbit_position)
                new_prey_state = rabbit.agent_step(rabbit_position, rabbit_action)
                env.preys[rabbit_id] = new_prey_state
                env.grid[new_prey_state[0], new_prey_state[1]] = 6

                # Calculate rabbit reward and update Q-table
                reward, prev_distance = rabbit.get_rabbit_reward(predator_state, new_prey_state, prev_distance)
                rabbit.update_rabbit_qtable(predator_state, rabbit_action, reward, new_prey_state, rabbit_position)

                # Remove rabbit if caught
                if math.dist(predator_state, new_prey_state) == 0:
                    del env.preys[rabbit_id]

                # If no rabbits left, break loop
                if not env.preys:
                    break

            # If no rabbits left, end episode
            if not env.preys:
                break

            # Check that there are still rabbits left before choosing the fox's action
            if env.preys:
                # Fox chooses action based on predator and remaining rabbits
                fox_action = fox.choose_fox_action(predator_state, env.preys)

                # Move fox
                env.grid[predator_state[0], predator_state[1]] = 0
                new_predator_state = fox.agent_step(predator_state, fox_action)
                env.grid[new_predator_state[0], new_predator_state[1]] = 1

                # Calculate fox reward and update Q-table
                reward, prev_distance = fox.get_fox_reward(new_predator_state, env.preys, prev_distance)               
                fox.update_fox_qtable(predator_state, fox_action, reward, new_predator_state, env.preys)

                # Check if fox caught any rabbit right after moving
                for rabbit_id, rabbit_position in list(env.preys.items()):
                    if new_predator_state == rabbit_position:  # Fox catches rabbit
                        del env.preys[rabbit_id]  # Remove rabbit from the environment
                        rabbits_caught_steps[rabbit_id].append(step + 1)  # Log catch step
                        rabbits_caught += 1
                        rabbits_left_at_episode -=1

                predator_state = new_predator_state  # Update predator state

                #if episode % 1000 == 0:
                # Visualise the grid every x steps
                    #env.visualise_grid_dynamic(episode)

                if new_predator_state in env.preys.values():
                    catch_steps.append((episode, step + 1))
                    rabbits_caught_steps[rabbit_id].append(step + 1)
                    rabbits_caught += 1
                elif step == max_steps - 1:
                    catch_steps.append((episode, step + 1))

                # End episode if all rabbits are caught
                if rabbits_caught == num_rabbits:
                    break


        # At the end of each episode record numbeer of rabbits remaining
        rabbits_left_record.append(len(env.preys)) 

        




# Start learning loop
plt.ioff()



qlearning_loop(episodes=500, max_steps=250, rabbits_caught=rabbits_caught)


print(rabbits_left_record)


# extract episode numbers and steps
episodes = [x[0] for x in catch_steps]
steps_to_catch = [x[1] for x in catch_steps]

plt.plot(rabbits_left_record)
plt.xlabel("Episodes")
plt.ylabel("Rabbits Left at End of Episode")
plt.title(f"Rabbits Remaining At Episode End \n Gridsize = {g_size} : Number of Rabbits: {num_rabbits}")
plt.show()

