#!/bin/sh

# Using a bash script to run multiple python codes.
# This is a command line instructions with the following elements:
# program -- file to run --  parameters n_cells and n_spots   -- output file

# if statement 

python3 ./simulation_tracking.py 50 35 >> out.txt
python3 ./simulation_tracking.py 50 40 >> out.txt
python3 ./simulation_tracking.py 50 33 >> out.txt

