#!/bin/bash
#SBATCH --job-name=swinade     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=swinade%j.out # output file name
#SBATCH --error=swinade%j.err  # error file name

set -x

cd $WORK/transseg2d

module purge
module load cuda/10.1.2

CONFIG="configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py"
GPUS=1
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python testslurm.py
