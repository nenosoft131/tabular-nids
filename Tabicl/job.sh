#!/bin/bash
#SBATCH -J ICL
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

source Tabicl/tabicl/myenv/bin/activate

#  python3.6 code/TaBERT/t.py
python3.11 Tabicl/embedding_gen.py
# python3.6 code/TaBERT/com.py

