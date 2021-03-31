#!/bin/sh

# Using a bash script to run multiple python codes.
# This is a command line instructions with the following elements:
# program -- file to run --  parameters n_cells and n_spots   -- output file
python3 ./simulation_tracking.py 2 35 >> out.txt
python3 ./simulation_tracking.py 2 33 >> out.txt

