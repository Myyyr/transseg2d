#!/bin/bash
#SBATCH --job-name=swin_tiny_ade     # job name
#SBATCH --ntasks=8                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4                 # nombre de GPU par nœud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=50:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/swin_tiny_ade%j.out # output file name
#SBATCH --error=logs/swin_tiny_ade%j.err  # error file name

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2



CONFIG="configs/orininal_swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py"
GPUS=8
PORT=${PORT:-29500}


srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3}
