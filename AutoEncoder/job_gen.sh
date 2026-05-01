#!/bin/bash
#SBATCH -J AUTOE_GEN       # Job name
#SBATCH -p standard                # Use the 'standard' GPU partition
#SBATCH --gres=gpu:L40:1          # Request 1 NVIDIA L40 GPU
#SBATCH -c 4                      # 4 CPU cores
#SBATCH --mem=64G                 # 16 GB of RAM
#SBATCH --time=24:00:00
#SBATCH --tmp=30G                 # 10 GB of temporary space
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com

echo "Job running on $(hostname)"
nvidia-smi                       # Check GPU status on the node

cd /home/user_name/work

source /home/user_name/miniconda3/etc/profile.d/conda.sh
conda activate tabert-py36

python3.6 AutoEncoder/gen_rep.py
