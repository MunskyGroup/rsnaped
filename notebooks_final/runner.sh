#!/bin/sh

# Bash script to run multiple python codes.
# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>>

# ########### ACTIVATE ENV #############################
# To load the env pass the specific location of the env and then activate it. 
# If not sure about the env location use: source activate <<venv_name>>   echo $CONDA_PREFIX
#source /home/"$USER"/anaconda3/envs/FISH_processing
conda activate rsnaped_env
export CUDA_VISIBLE_DEVICES=0,1

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name  of the <<python_file.py>>, and  the rest are in positional order 
# Make sure to convert str to the desired data types.

# Paths with configuration files
path_to_executable="${PWD%/*}/notebooks_final/testing_conditions.py" 
nohup python3 "$path_to_executable" >> 'output.txt' &

# #####################################
# #####################################

# Deactivating the environment
conda deactivate

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner.sh /dev/null 2>&1 & disown

# ########### TO MONITOR PROGRESS #########################
# To check if the process is still running
# ps -ef | grep python3
# ps -ef | grep python3 | grep "testing_conditions"
# ps -ef | grep python3 | grep "testing_conditions" | awk '{print $2}'   # Processes running the pipeline.
# kill $(ps -ef | grep python3 | grep "testing_conditions" | awk '{print $2}')

# nvidia-smi | grep 'Default'

# To remove files
# ls *.tif
# ls *temp_ out*
# rm *.tif out*
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* output__* temp_* *.tif

exit 0