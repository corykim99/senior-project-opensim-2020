#!/bin/bash
srun -N1 -n1 -p gpu --exclusive python q_train.py
exit 0