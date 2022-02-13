#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=amdlong         # put the job into the gpu partition/queue
#SBATCH --output=log.out     # file name for stdout/stderr
#SBATCH --mem=50G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=3-00:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=datagen        # job name (default is the name of this file)

cd $HOME/lidar/
#  poustej pres for loop s ampersandem a logovanim

python -u cool_model/generate_data.py
#wait


