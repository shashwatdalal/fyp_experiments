import os

JOB_NAME = 'hello-cluster-world'

config = """#!/bin/bash
#SBATCH --gres=gpu:1,gpu2
#SBATCH --mail-type=NONE # required to send email notifcations
#SBATCH --output={}
#SBATCH --job-name={}
export PATH=/vol/bitbucket/${{USER}}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
python model.py
/usr/bin/nvidia-smi
uptime""".format("{}.out".format(JOB_NAME), JOB_NAME)

CONFIG_FILE = 'job_sumbission.sh'
if os.path.exists(CONFIG_FILE):
  os.remove(CONFIG_FILE)
with open(CONFIG_FILE, 'w+') as cf_fd:
    cf_fd.write(config)