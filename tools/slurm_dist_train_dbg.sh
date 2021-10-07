#!/usr/bin/env bash
#SBATCH --time=00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/swin_tiny_ade%j.out # output file name
#SBATCH --error=logs/swin_tiny_ade%j.err  # error file name
#SBATCH --job-name=swin_tiny_ade     # job name

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2



CONFIG="configs/orininal_swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py"
GPUS=$1
GPUS_PER_NODE=$2
CPUS_PER_TASK=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    python -u tools/train.py ${CONFIG} --launcher="slurm"
