#!/bin/sh
#$ -cwd
#$ -N rsnaped_job
#$ -o output_rsnaped.txt
#$ -e error_rsnaped.err
#$ -q munsky.q@node*

/usr/local/anaconda3/bin/python3 simulation_tracking.py '$5'