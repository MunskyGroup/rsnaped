#!/bin/sh

# Using a bash script to run multiple python codes.
# This is a command line instructions with the following elements:
# program -- file to run --  parameters n_cells and n_spots   -- output file

#    total_number_of_spots 
#    simulation_time 
#    ke_gene_0
#    ke_gene_1
#    ki_gene_0 
#    ki_gene_1

# python3 ./data_ml_bash.py total_number_of_spots simulation_time  ke_gene_0 ke_gene_1 ki_gene_0 ke_gene_1  >> out.txt
python3 ./data_ml_bash.py 100 30 10 10 0.03 0.03  >> out.txt

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner.sh /dev/null 2>&1 & disown

# ########### TO MONITOR PROGRESS #########################
# To check if the process is still running
# ps -ef | grep python3
# ps -ef | grep python3 | grep "pipeline_"
# ps -ef | grep "python3 ./pipeline_executable.py *" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep "python3 ./pipeline_executable.py *" | awk '{print $2}')

# nvidia-smi | grep 'Default'
# top -u luisub | grep 'python3'

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* output__* temp_* *.tif