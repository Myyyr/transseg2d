#!/bin/bash
#SBATCH --job-name=swin_tiny_ade     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/swin_tiny_ade%j.out # output file name
#SBATCH --error=logs/swin_tiny_ade%j.err  # error file name

set -x

cd $WORK/transseg2d
module purge
module load cuda/10.1.2

CONFIG="configs/orininal_swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py"
GPUS=1
PORT=${PORT:-29500}

PYTHONPATH="tools/..":$PYTHONPATH \
srun python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3}
