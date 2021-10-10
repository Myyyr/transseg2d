#!/bin/bash
#SBATCH --job-name=swinunet_tiny_ade_pt     # job name
#SBATCH --ntasks=8                  # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/swinunet_tiny_ade_pt%j.out # output file name
#SBATCH --error=logs/swinunet_tiny_ade_pt%j.err  # error file name

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2



# CONFIG="configs/orininal_swin/upernet_swin_tiny_pt_patch4_window7_512x512_160k_ade20k.py"
CONFIG="configs/swinunet/swinunet_tiny_patch4_window7_512x512_160k_ade20k.py"
RESUME="work_dirs/swinunet_tiny_patch4_window7_512x512_160k_ade20k/128000.pth"

# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --options model.pretrained="pretrained_models/swin_tiny_patch4_window7_224.pth" --launcher="slurm" ${@:3}
srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --resume-from=$RESUME --launcher="slurm" ${@:3}
