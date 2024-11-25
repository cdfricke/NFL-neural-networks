#!/bin/bash

#PBS -N FCNTRAINING
#PBS -A PAS2038
#PBS -l walltime=3:00:00
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -j oe
#
# Makes sure we use the correct python
module reset
#
# NOTE: next line is for bash only (ask if you use c-shell)
source /fs/ess/PAS2038/PHYSICS_5680_OSU/jupyter/bin/activate
which python
module load cuda/11.2.2
#
cd $HOME/osc_classes/PHYSICS_5680_OSU/NFL-neural-networks/scripts || exit
python -u FCN_training.py
