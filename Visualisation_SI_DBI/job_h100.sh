#!/bin/bash
#SBATCH -J VIS
#SBATCH -p h100                  # ✅ Correct partition for jnultra02
#SBATCH --gres=gpu:H100:2
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --tmp=30G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com

echo "Job running on $(hostname)"
nvidia-smi

cd /home/s445773/work/Vis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rapidsfinal

/home/s445773/miniconda3/envs/rapidsfinal/bin/python vis.py
