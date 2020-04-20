#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE # required to send email notifcations
#SBATCH --mail-user=<your_username> # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output test.out
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
python model.py
/usr/bin/nvidia-smi
uptime
