# Group 3
# ECE4317
# Project
# Q Learning Agent

import numpy as np
import random
import gym
import math
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from osim.env import L2M2019Env
import sys
import h5py

# Show entire qtable
np.set_printoptions(threshold=sys.maxsize)

# Adjust max_episode_steps and episodes for different learning

# Initialize Environment
env_name = 'L2M2019Env'
env = L2M2019Env(visualize=False)
env.reset()
env._max_episode_steps = 10 #set max steps per episode
#env.seed(0) #set environment seed for same initial positions
#np.random.seed(0) #set numpy rng to reproduce same "random" action sequence

# Get State Space
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# Set Hyperparameters
initial_lr = 1.0 #learning rate
min_lr = 0.005 #min learning rate
gamma = 0.8 #discount factor = balances immediate and future reward (ranges 0.8 to 0.99)
epsilon = 0.05 #higher -> more exploitation, less exploration
n_states = 339 #number of states
n_action = 2**env.action_space.shape[0]
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
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    
    # retrieve actual observation values and clean array
    # v_tgt_field: 242 observation values
    # pelvis: 9 observation values
    # r_leg: 44 observation values
    # l_leg: 44 observation values
    # total: 339 observation values
    obs_val = []
    index = 0
    for key, value in recursive_items(obs):
        if isinstance(value, (list, np.ndarray)):
            for i in value:
                if isinstance(i, (list, np.ndarray)):
                    for j in i:
                        if isinstance(j, (list, np.ndarray)):
                            for k in j:
                                if k > env_high[index]:
                                    obs_val.append(env_high[index])
                                elif k < env_low[index]:
                                    obs_val.append(env_low[index])
                                else:
                                    obs_val.append(k)
                                index += 1
                        else:
                            if j > env_high[index]:
                                obs_val.append(env_high[index])
                            elif j < env_low[index]:
                                obs_val.append(env_low[index])
                            else:
                                obs_val.append(j)
                            index += 1
                else:
                    if i > env_high[index]:
                        obs_val.append(env_high[index])
                    elif i < env_low[index]:
                        obs_val.append(env_low[index])
                    else:
                        obs_val.append(i)
                    index += 1
        else:
            if value > env_high[index]:
                obs_val.append(env_high[index])
            elif value < env_low[index]:
                obs_val.append(env_low[index])
            else:
                obs_val.append(value)
            index += 1
    
    # Obtain density values (step sizes)
    env_den = np.subtract(env_high, env_low)
    env_den = np.divide(env_den, n_states)

    # Scale values
    obs_scaled = np.subtract(obs_val, env_low)
    obs_scaled = np.divide(obs_scaled, env_den)
    for i in range(339):
        if obs_scaled[i] > 0:
            obs_scaled[i] = obs_scaled[i] - 1
    obs_scaled = obs_scaled.astype(int)
    #print(obs_scaled)
    return obs_scaled

try:
    # Load trained data
    print("Loading QTable")
    #q_table = np.load('train_data/osim_q_table.npy')
    q_table = h5py.File('./train_data/osim_q_table.h5', 'r')
    print("QTable has been loaded")
    
except OSError:
    # fill Q-table with zeros
    print("Generating zeros")
    zeros = np.zeros((n_states, n_action)) #fill Q-table with zeros
    
    # Create mc_train_time.csv
    with open('./train_data/osim_train_time.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['Steps', 'Episodes', 'Time Elapsed'])
    
    # Create mc_reward.csv
    with open('./train_data/osim_reward.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(['Reward'])
        
    # Create mc_alpha.csv
    with open('./train_data/osim_alpha.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(['Alpha'])

    # Q-table = 3D table
    # Rows = states (states = 2D table : pos, vel)
    # Columns = actions
    # look into exporting data
    q_table = zeros

    # Save Q-table as mc_q_table.csv
    #np.save('train_data/osim_q_table.npy', q_table)
    h5py.File('./train_data/osim_q_table.h5', 'w')
    print("New QTable has been generated")

# Store Training Start Time
time_start = time.time()

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
        #env.render()
        obs_val = discretization(env, obs)
        
        # action based on current state using epsilon greedy
        # Epsilon Greedy - balances exploration and exploitation by choosing them randomly
        # Exploration - perform new actions, riskier, slow but improve accuracy in long-term
        # Exploitation - perform similar actions that gave high rewards, safer, fast but may produce inaccurate results
        if np.random.uniform(low = 0, high = 1) < epsilon:
            #print("Explore")
            # Explore
            # Binary a -> Decimal i
            a = np.random.randint(2, size=22)
            
            i = int("".join(str(x) for x in a), 2) 
        else:
            #print("Exploit")
            # Exploit
            # Decimal i -> Binary a
            i = int(np.argmax(q_table[obs_val]))
            #print(q_table[obs_val])
            
            a = [int(j) for j in bin(i)[2:]]
            a = np.array(a)
            if len(a) < 22:
                a = np.pad(a, (22-len(a), 0), 'constant')
        
        obs, reward, terminate, _ = env.step(a)
        total_reward += reward
        
        # Update Q-table
        obs_val_ = discretization(env, obs)
        obs_val = obs_val[:,None]
        q_table[obs_val, i] = (1 - alpha) * q_table[obs_val, i]  + alpha * (reward + gamma * np.max(q_table[obs_val_]))
        steps += 1
        
        # Goal reach (cart reached flag)
        if terminate:
            break
        elif episode % 100 == 0:
            # Save Rewards
            with open('./train_data/osim_reward.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow(list_reward)

            # Save Alpha
            with open('./train_data/osim_alpha.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow(list_alpha)

            # Save Q-table
            #np.save('train_data/osim_q_table.npy', q_table)
            h5py.File('./train_data/osim_q_table.h5', 'w')

            # Store Training End Time
            time_end = time.time()
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
            with open('./train_data/osim_train_time.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow([str(env._max_episode_steps), str(episodes), time_elapsed])
            
    # Print results when episode is complete
    print("Episode : Total Reward : Steps\t\t{} : {} : {}".format(episode + 1, truncate(total_reward, 3), steps))
    #print(q_table)
    # Add rewards and alpha to list
    list_reward.append(total_reward)
    list_alpha.append(alpha)

# Save Rewards
with open('./train_data/osim_reward.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(list_reward)

# Save Alpha
with open('./train_data/osim_alpha.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(list_alpha)

# Save Q-table
#np.save('train_data/osim_q_table.npy', q_table)
h5py.File('./train_data/osim_q_table.h5', 'w')

# Store Training End Time
time_end = time.time()
time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
with open('./train_data/osim_train_time.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow([str(env._max_episode_steps), str(episodes), time_elapsed])

# Output training times
print("Training Complete\nTime elapsed: {}".format(time_elapsed))