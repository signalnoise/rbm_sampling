#! /bin/sh
#$ -cwd
#$ -N thermodynamics
#$ -l h_rt=04:00:00
#$ -pe gpu 1
#$ -l h_vmem=16G

. /etc/profile.d/modules.sh

module load igmm/apps/anaconda/5.0.0.1
source activate owen-pytorch
python3 thermodynamics_convergence.py --cuda True --json thermodynamics.json --training_data ../rbm-pytorch-refactor/training_data/state$SGE_TASK_ID.txt --input_path ../rbm-pytorch-refactor/data/$SGE_TASK_ID  --output_path ../rbm-pytorch-refactor/data/$SGE_TASK_ID