#! /bin/sh
#$ -cwd
#$ -N hists
#$ -l h_rt=04:00:00
#$ -pe gpu 1
#$ -l h_vmem=16G

. /etc/profile.d/modules.sh

module load igmm/apps/anaconda/5.0.0.1
source activate owen-pytorch
python3 plotter.py --json thermodynamics-params.json