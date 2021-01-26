import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import gym

env = gym.make("Pendulum-v0")
print("Pendulum-v0")
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

cos, sin, dot = env.reset()
state = env.reset()

print(state)