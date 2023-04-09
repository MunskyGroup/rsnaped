#!/bin/bash
# #SBATCH --gres=gpu:4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=12
# #SBATCH --nodelist=node28    # gpu2 gpu3 gpu4
#SBATCH --partition=all
# #SBATCH --ntasks=4
#SBATCH --job-name=sim_par

# module purge
module load gnu9/9.4.0 
module load cudnn/8.3-10.2

# If needed, use this to change file permissions -> chmod 755 <<script_name.sh>

# ########### PROGRAM ARGUMENTS #############################
# If the program requieres positional arguments. 
# Read them in the python file using: sys.argv. This return a list of strings. 
# Where sys.argv[0] is the name of the <<python_file.py>>, and  the rest are in positional order 

####################  PATHS TO CODE FILES  ############################

path_to_executable="${PWD%/*}/notebooks_final/running_conditions.py" 
~/.conda/envs/rsnaped_env/bin/python "$path_to_executable"

# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: sbatch runner_cluster.sh /dev/null 2>&1 & disown

exit 0

# ########### TO REMOVE SOME FILES #########################

# To remove files
# ls *.tif
# rm -r temp_*
# rm -r analysis_*
# rm -r slurm* out* temp_* masks_* 

# ########### SLURM COMMANDS #########################
# scancel [jobid]
# squeue -u [username]
# squeue

# ########### CLUSTER INSTALLATION #########################
# git clone --depth 1 https://github.com/MunskyGroup/rsnaped.git
# /opt/ohpc/pub/apps/anaconda3/bin/conda env remove -n rsnaped_env
# /opt/ohpc/pub/apps/anaconda3/bin/conda env create -f rsnaped_env.yml