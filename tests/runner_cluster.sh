#!/bin/bash
#$ -cwd
#$ -N j1
#$ -o output_rsnaped.txt
#$ -e error_rsnaped.er 
# #$ -q munsky.q@node*
#$ -q munsky-gpu.q@gpu*

# module purge
# module load apps/anaconda3
# module load conda/10.0

source /home/students/luisub/.conda/envs/t0/bin/activate

# /home/students/luisub/.conda/envs/rsnaped_env/bin/python3 ./simulation_tracking.py 2 >> out.txt
~/.conda/envs/t0/bin/python ./simulation_tracking.py 10 30 >> out.txt