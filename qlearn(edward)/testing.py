# Senior Project
# OpenSimRL Q Learning Agent
# Edward's test

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime
from osim.env import L2M2019Env
import sys

#Show full qtable
#np.set_printoptions(threshold=sys.maxsize)


# Adjust max_episode_steps and episodes for different learning

# Initialize Environment
env_name = 'L2M2019Env'
env = L2M2019Env(visualize=False) #True)
env.reset()
env._max_episode_steps = 1000 #set max steps per episode
env.seed(0) #set environment seed for same initial positions
np.random.seed(0) #set numpy rng to reproduce same "random" action sequence

# Get State Space
#print("Action Space {}".format(env.action_space))
#print("State Space {}".format(env.observation_space))

print("Action Space: ", env.get_action_space_size())
print("State Space: ", env.get_observation_space_size())

action_size = int(env.get_action_space_size())
state_size = int(env.get_observation_space_size())

#print("Action size int: ", action_size)
#print("State size int: ", state_size)

# Set Hyperparameters
initial_lr = 1.0 #learning rate
min_lr = 0.005 #min learning rate
gamma = 0.8 #discount factor = balances immediate and future reward (ranges 0.8 to 0.99)
#epsilon = 0.05 
epsilon = 1.0 
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001
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
    for i in range(state_size):
        if obs_scaled[i] > 0:
            obs_scaled[i] = obs_scaled[i] - 1
    obs_scaled = obs_scaled.astype(int)
    #print(obs_scaled.size)
    return obs_scaled

# Q-table = 3D table
# Rows = states (states = 2D table : pos, vel)
# Columns = actions
# look into exporting data
# 339 x 22
#q_table = np.zeros((n_states, env.action_space.shape[0])) #fill Q-table with zeros
q_table = np.zeros((state_size, action_size))

# Store Training Start Time
now = datetime.now()
time_start = now.strftime("%H:%M:%S")

for episode in range(episodes):
    
    # Initialize environment for this episode
    obs = env.reset() #random state
    #print("Random state :", obs)
    total_reward = env._max_episode_steps

    # decreasing learning rate (alpha) over time
    # reduce steps when approaching goal, prevent overstep
    alpha = max(min_lr, initial_lr * (gamma**(episode//100))) #learning rate
    steps = 0 #initialize steps
    
    # Agent learning
    while True:
        #env.render()
        obs_val = discretization(env, obs)
        #print("Current random obs val:", obs_val)
        # action based on current state using epsilon greedy
        # Epsilon Greedy - balances exploration and exploitation by choosing them randomly
        # Exploration - perform new actions, riskier, slow but improve accuracy in long-term
        # Exploitation - perform similar actions that gave high rewards, safer, fast but may produce inaccurate results
        #if np.random.uniform(low = 0, high = 1) < epsilon:
        exp_exp_tradeoff = random.uniform(0,1)
        if exp_exp_tradeoff > epsilon: #If random number > epsilon then take best action, exploit
            # Exploit
            #a = np.random.randint(2, size=22)
            #i = int(np.argmax(q_table[obs_val]) / 22) #fix half qtable not filling
            #i = int(np.argmax(obs_val))
            #print("Index:", i)
            #print("Qtable shape:", q_table.size)
            #print(q_table[obs_val])
            #a = np.array(q_table[obs_val[i]]).astype(int)

            #find index, from array size 22, that has the largest qvalue
            i = np.argmax(q_table[obs_val,:])
            print("Index of largest qvalue: ", i)

            a = q_table[obs_val[i]]
            #print("Exploit best action: ")
            #print(a)            
            a = np.around(a).astype(int)
            #print("Exploit best action rounded: ")
            #print(a)
            #print("qtable obs val:", q_table[obs_val[i]])
        else:
            # Explore
            a = np.random.randint(2, size=22)

            '''
            a = np.random.uniform(0,1,22)
            for i in range(22):
                a[i] = truncate(a[i], 1)
                
            a.astype(int)
            '''
            #print("Exploring...")
            #print("Exploration random table: ")
            #print(a)
        
        
        
        #obs, reward, terminate, _ = env.step(a)
        #print("Before stepping a:", a)
        new_obs, reward, terminate, info = env.step(a)
        new_obs_ = discretization(env, obs)
        total_reward += reward
        
        # Update Q-table
        #print("Action shape: ", a.shape)
        a_dim = a[None,:]
        #print("A: none", a_dim.shape)
        
        #print("Obs val shape: ", obs_val.shape)
        obs_val_dim = obs_val[:, None]
        #print("Obs val: none", obs_val_dim.shape)        
              
        q_table[obs_val_dim, a_dim] = q_table[obs_val_dim, a_dim] + alpha * (reward + gamma * np.max(q_table[new_obs_, :]) - q_table[obs_val_dim, a_dim])
        '''
        obs_val_ = discretization(env, obs)
        a = a[None,:]
        print("A: none")
        print(a)
        obs_val = obs_val[:,None]
        #q_table[obs_val, a] = (1 - alpha) * q_table[obs_val, a]  + alpha * (reward + gamma * np.max(q_table[obs_val_]))
        #print(reward)
        #print(q_table)
        #print("(1 - {}) * {} + {} * ({} + {} * {})".format(alpha, q_table[obs_val, a], alpha, reward, gamma, np.max(q_table[obs_val_])))
        '''
        
        steps += 1
        
        
        if terminate:
            break
        
        #reducing epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
            
    # Print results when episode is complete
    print("Episode : Total Reward : Steps\t\t{} : {} : {}".format(episode + 1, truncate(total_reward, 3), steps))
    print(q_table)
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