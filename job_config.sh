#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE # required to send email notifcations
#SBATCH --output=reddit_comments_100_test.out
#SBATCH --job-name=reddit_comments_100_test
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
python main.py
/usr/bin/nvidia-smi
uptime