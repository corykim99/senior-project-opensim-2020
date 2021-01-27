#!/bin/bash
#SBATCH --mem=64GB
srun -N 1 -n 1 -p gpu python q_train.py
exit 0