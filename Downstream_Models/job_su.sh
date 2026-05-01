#!/bin/bash
#SBATCH -J SO_SUPER
#SBATCH -p standard                # Use the 'standard' GPU partition
#SBATCH --gres=gpu:L40:2          # Request 1 NVIDIA L40 GPU
#SBATCH -c 4                      # 4 CPU cores
#SBATCH --mem=64G                 # 16 GB of RAM
#SBATCH --time=24:00:00
#SBATCH --tmp=30G                 # 10 GB of temporary space
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com

echo "Job running on $(hostname)"
nvidia-smi                       # Check GPU status on the node

cd /home/s445773/work/GuKNN
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rapidsfinal

#  python3.6 code/TaBERT/t.py
/home/s445773/miniconda3/envs/rapidsfinal/bin/python sota_super.py
# python3.6 code/TaBERT/com.py

