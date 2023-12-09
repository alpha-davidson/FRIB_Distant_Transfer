#!/bin/bash
### job parameters ###
#SBATCH --job-name "Distant Transfer"
#SBATCH --mem 32G
#SBATCH --gpus 1

# Activate `Your Conda Env Name` conda environment
source /opt/conda/bin/activate aot

# Run the script for Data_Sampling
# The purpose of this stage is to convert data from h5 format to 
# numpy array, and also to remove junk data (data with less than 5 points) 
# and sample the data.
# python Data_Sampling.py

# Run the script for Prediction.py
python Prediction.py