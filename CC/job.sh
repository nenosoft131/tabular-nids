#!/bin/bash
#SBATCH -J CC
#SBATCH -p h100                  # ✅ Correct partition for jnultra02
#SBATCH --gres=gpu:H100:2
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --tmp=30G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
echo "Job running on $(hostname)"
nvidia-smi                       # Check GPU status on the node

cd /home/user_name/work

source CC/myenv/bin/activate

#  python3.6 code/TaBERT/t.py
python3.11 CC/run.py
# python3.6 code/TaBERT/com.py

