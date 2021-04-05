#!/bin/bash
#$ -cwd
#$ -N rsnaped_job
#$ -e error_rsnaped.er
#$ -o test_o.txt 
#$ -q munsky-gpu.q@gpu*

# module purge
# module load apps/anaconda3
# module load conda/10.0

#source /home/students/luisub/.conda/envs/t0/bin/activate
/home/students/bsilagy/.conda/envs/rsnaped_env/bin/python3 ./simulation_tracking.py 20 40 >> out.txt
#python simulation_tracking.py 20 40 >> out.txt
