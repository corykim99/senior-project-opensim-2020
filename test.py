from osim.env import L2M2019Env
import tensorflow as tf
from tensorflow.keras import layers
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

#print(state)
print("v_tgt_field: " + v_tgt_field)
print("pelvis: " + pelvis)
print("r_leg: " + r_leg)
print("l_leg: " + l_leg)
#print(state)

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

#tf_state = tf.expand_dims(tf.convert_to_tensor(state["v_tgt_field"]), 0)

#print(env.observation_space.shape[0])

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