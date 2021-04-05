#!/bin/bash
#$ -cwd
#$ -N rsnaped_job
#$ -e error_rsnaped.er 
#$ -q munsky-gpu.q@gpu*

# module purge
# module load apps/anaconda3
# module load conda/10.0

#source /home/students/luisub/.conda/envs/t0/bin/activate
/home/students/luisub/.conda/envs/rsnaped_env/bin/python3 ./simulation_tracking.py >> out.txt
#python simulation_tracking.py 20 40 >> out.txt
