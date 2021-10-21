#!/bin/bash
#SBATCH --job-name=s24g5baade     # job name
#SBATCH --ntasks=8                  # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=30:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/s24g5baade%j.out # output file name
#SBATCH --error=logs/s24g5baade%j.err  # error file name

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2


# RESUME="work_dirs/swinunet_tiny_patch4_window7_512x512_160k_ade20k/iter_128000.pth"
# RESUME="work_dirs/swinunetgtv1_tiny_patch4_window7_512x512_160k_ade20k/iter_32000.pth"
# RESUME="work_dirs/swinunetgtv2_tiny_patch4_window7_512x512_160k_ade20k/iter_32000.pth"

# CONFIG="configs/orininal_swin/upernet_swin_tiny_pt_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunet/swinunet_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunet/swinunet_tiny_patch4_window7_512x512_160k_ade20k_lr.py"
# CONFIG="configs/swinunet/swinunet_tiny_patch4_window7_512x512_160k_ade20k_ptd.py"
# CONFIG="configs/noswinunet/noswinunet_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetgtv1/swinunetgtv1_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetgtv2/swinunetgtv2_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetgtv2/swinunetgtv2_tiny_patch4_window7_512x512_160k_ade20k_t10.py"

# CONFIG="configs/swinunetv2/swinunetv2_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv3/swinunetv2gtv3_g10_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv3/swinunetv2gtv3_g5_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv3/swinunetv2gtv3_g20_tiny_patch4_window7_512x512_160k_ade20k.py"

# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g1_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g10_tiny_patch4_window7_512x512_160k_ade20k.py"

# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_small_patch4_window7_512x512_160k_ade20k.py"
CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_base_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_tiny_patch4_window7_769x769_160k_cityscapes.py"
# CONFIG="configs/swinunetv2gtv3/swinunetv2gtv3_g5_tiny_patch4_window7_769x769_160k_cityscapes.py"

# PRET="pretrained_models/swin_tiny_patch4_window7_224.pth"
# PRET="pretrained_models/swin_small_patch4_window7_224.pth"
PRET="pretrained_models/swin_base_patch4_window7_224.pth"
# PRET="pretrained_models/swin_base_patch4_window7_224_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window12_384_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window12_384.pth"



srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --options model.pretrained=$PRET --launcher="slurm" ${@:3}
# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --resume-from=$RESUME --launcher="slurm" ${@:3}
