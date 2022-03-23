#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=16GB
#SBATCH --chdir=/home/gridsan/scasper/multiagent-competition/outfiles
#SBATCH --job-name=tmp
#SBATCH --gres=gpu:volta:1

cd
module load anaconda/2022a
module load cuda/10.0
source activate mac
conda activate mac
rm -r -f /state/partition1/user/scasper/
mkdir /state/partition1/user/scasper/
cd /state/partition1/user/scasper/
cp -r ~/mujoco-py .
cp -r ~/multiagent-competition .
cd mujoco-py
python setup.py install
python -c 'import mujoco_py'
cp -r mujoco_py ../multiagent-competition
cd ../multiagent-competition
python -c 'import mujoco_py'



