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

# Bin to Dec
a = np.random.randint(2, size=22)
i = int("".join(str(x) for x in a), 2) 

#print(i)

# Dec to Bin
res = [int(j) for j in bin(i)[2:]]
res = np.array(res)
res = np.pad(res, (22-len(res), 0), 'constant')