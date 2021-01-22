from osim.env import L2M2019Env
import numpy as np
import pandas as pd
import gym

env = L2M2019Env(visualize=True)
observation = env.reset()

# Get State Space
print("L2M2019Env")
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

'''
env_p = gym.make("Pendulum-v0")
print("Pendulum-v0")
print("Action Space {}".format(env_p.action_space))
print("State Space {}".format(env_p.observation_space))
'''
'''
# Action Space
# First 11 = Right Leg
# Last 11 = Left Leg
# 0 = relax muscle
move = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
count = 0

for i in range(1000):
    if i < 50:
        move = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif i == 51:
        move = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        if count == 25:
            for j in range(22):
                if move[j] == 1:
                    move[j] = 0
                else:
                    move[j] = 1
            count = 0
        count += 1
    observation, reward, done, info = env.step(move)
'''

# state: map
# Keys: v_tgt_field, pelvis, r_leg, l_leg
# Values: action_space
v_tgt_field, pelvis, r_leg, l_leg = env.reset()
state = env.reset()
'''
#print(state)
print("v_tgt_field: " + v_tgt_field)
print("pelvis: " + pelvis)
print("r_leg: " + r_leg)
print("l_leg: " + l_leg)
#print(state)
'''
# v_tgt_field: 2x11x11 array containing action space values
#print("{}: {}\n\n".format("v_tgt_field", state["v_tgt_field"]))
#print(state["v_tgt_field"][1,10])
#print("\n\n")

# pelvis: map
# Keys: height, pitch, roll, vel
# Values: float, float, float, array size 6
#print("{}: {}\n\n".format("pelvis", state["pelvis"]))
#print(state["pelvis"]["height"])
#print("\n\n")

# r_leg: map
# Keys: each muscle in the right leg
# Values: 
#print("{}: {}\n\n".format("r_leg", state["r_leg"]))

# l_leg: map
# Keys: each muscle in the left leg
# Values: 
#print("{}: {}\n\n".format("l_leg", state["l_leg"]))

#print(env.observation_space.shape[0])

'''
def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)
muscles = []
for key, value in recursive_items(state):
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

print(muscles)
'''
'''
env_low = env.observation_space.low
env_high = env.observation_space.high
print("Env LOW: Size {} \n{}".format(len(env_low), env_low))
print("Env HIGH: Size {} \n{}".format(len(env_high), env_high))
'''
'''
print("Action Space")
print(env.action_space.n)
'''
def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)

# Obtain min and max values
env_low = env.observation_space.low
env_high = env.observation_space.high

# retrieve actual observation values and clean array
# v_tgt_field: 242 observation values
# pelvis: 9 observation values
# r_leg: 44 observation values
# l_leg: 44 observation values
# total: 339 observation values
n_states = 3390
obs_val = []
index = 0
for key, value in recursive_items(state):
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

print(obs_val)

# Obtain density values (step sizes)
env_den = np.subtract(env_high, env_low)
env_den = np.divide(env_den, n_states)

# Scale values
obs_scaled = np.subtract(obs_val, env_low)
print(obs_scaled)
obs_scaled = np.divide(obs_scaled, env_den)
print(obs_scaled)
obs_scaled = obs_scaled.astype(int)
print(obs_scaled)