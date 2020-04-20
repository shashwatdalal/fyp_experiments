#!/bin/bash
#SBATCH --gres=gpu:1,gpu2
#SBATCH --mail-type=NONE # required to send email notifcations
#SBATCH --output=hello-cluster-world2.out
#SBATCH --job-name=hello-cluster-world2
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
python model.py
/usr/bin/nvidia-smi
uptime