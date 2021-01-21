# Group 3
# ECE4317
# Project
# Q Learning Agent

import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
from datetime import datetime
from osim.env import L2M2019Env

# Adjust max_episode_steps and episodes for different learning

# Initialize Environment
env_name = 'L2M2019Env'
env = L2M2019Env(visualize=True)
env.reset()
env.render()
env._max_episode_steps = 1000 #set max steps per episode
env.seed(0) #set environment seed for same initial cart positions
np.random.seed(0) #set numpy rng to reproduce same number sequence

# Get State Space
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# Set Hyperparameters
initial_lr = 1.0 #learning rate
min_lr = 0.005 #min learning rate
gamma = 0.8 #discount factor = balances immediate and future reward (ranges 0.8 to 0.99)
epsilon = 0.05 #higher -> more exploitation, less exploration
n_states = 339 #number of states
episodes = 1000 #number of episodes

# List to track reward and alpha values
list_reward = []
list_alpha = []

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)

# Truncate total_reward decimal places
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

# Convert continuous (position and velocity) values into discrete values
def discretization(env, obs):
    
    # Obtain min and max values
    env_low = env.observation_space.low[0]
    env_high = env.observation_space.high[0]
    
    # v_tgt_field: 242 observation values
    # pelvis: 9 observation values
    # r_leg: 44 observation values
    # l_leg: 44 observation values
    # total: 339 observation values
    muscles = []
    for key, value in recursive_items(obs):
        if isinstance(value, (list, np.ndarray)):
            for i in value:
                if isinstance(i, (list, np.ndarray)):
                    for j in i:
                        if isinstance(j, (list, np.ndarray)):
                            for k in j:
                                muscles.append(k)
                        else:
                            muscles.append(j)
                else:
                    muscles.append(i)
        else:
            muscles.append(value)
    
    return muscles

# Q-table = 3D table
# Rows = states (states = 2D table : pos, vel)
# Columns = actions
# look into exporting data
q_table = np.zeros((n_states, n_states, env.action_space.shape[0])) #fill Q-table with zeros

# Store Training Start Time
now = datetime.now()
time_start = now.strftime("%H:%M:%S")

for episode in range(episodes):
    
    # Initialize environment for this episode
    obs = env.reset()
    total_reward = env._max_episode_steps
    
    # decreasing learning rate (alpha) over time
    # reduce steps when approaching goal, prevent overstep
    alpha = max(min_lr, initial_lr * (gamma**(episode//100))) #learning rate
    steps = 0 #initialize steps
    
    # Agent learning
    while True:
        env.render()
        muscles = discretization(env, obs)
        
        # action based on current state using epsilon greedy
        # Epsilon Greedy - balances exploration and exploitation by choosing them randomly
        # Exploration - perform new actions, riskier, slow but improve accuracy in long-term
        # Exploitation - perform similar actions that gave high rewards, safer, fast but may produce inaccurate results
        if np.random.uniform(low = 0, high = 1) < epsilon:
            # Explore
            a = np.random.choice(env.action_space.n)
        else:
            # Exploit
            a = np.argmax(q_table[muscles][muscles])
        
        obs, reward, terminate, _ = env.step(a)
        total_reward += reward
        
        # Update Q-table
        pos_, vel_ = discretization(env, obs)
        q_table[pos][vel][a] = (1 - alpha) * q_table[pos][vel][a] + alpha * (reward + gamma * np.max(q_table[pos_][vel_]))
        steps += 1
        
        # Goal reach (cart reached flag)
        if terminate:
            break
            
    # Print results when episode is complete
    print("Episode : Total Reward : Steps\t\t{} : {} : {}".format(episode + 1, truncate(total_reward, 3), steps))
    
    # Add rewards and alpha to list
    list_reward.append(total_reward)
    list_alpha.append(alpha)

# Store Training End Time
now = datetime.now()
time_end = now.strftime("%H:%M:%S")

# Output training times
print("Training Complete\nStart : End\t\t{} : {}".format(time_start, time_end))

# Initialize Range
x = range(episodes)

'''
# Graph results and total reward over all episodes
plt1 = plt.figure(1)
plt.plot(x, list_reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes')

# Graph the alpha over all episodes
plt2 = plt.figure(2)
plt.plot(x, list_alpha)
plt.xlabel('Episode')
plt.ylabel('Alpha')
plt.title('Alpha over Episodes')

# Output Plot
plt.show()
'''