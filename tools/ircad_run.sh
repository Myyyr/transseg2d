#!/bin/bash
#SBATCH --job-name=test     # job name
#SBATCH -p public
#SBATCH --gpus=4
#SBATCH --output=logs/test%j.out # output file name
#SBATCH --error=logs/test%j.err  # error file name

source /opt/server-env.sh 
module purge
module load anaconda/2019.07
# conda activate open-mmlab
# module load cuda/10.1


# ln -s /scratch/lthemyr/cityscapes data/



# ..............................................
# CONFIG="configs/orininal_swin/zz_upernet_swin_base_patch4_window7_769x769_160k_cityscapes.py"
CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_base_patch4_window7_769x769_160k_cityscapes.py"

# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_base_patch4_window7_512x512_160k_ade20k.py"

# ..............................................
# CONFIG="configs/swinunetv2/zz_swinunetv2_tiny_patch4_window7_769x769_80k_cityscapes_good.py"
# CONFIG="configs/swinunetv2/zz_swinunetv2_tiny_patch4_window7_769x769_160k_cityscapes_good.py"
# CONFIG="configs/swinunetv2gtv8/zz_swinunetv2gtv8_g10_tiny_patch4_window7_769x769_160k_cityscapes_good.py"
# CONFIG="configs/swinunetv2gtv8/zz_swinunetv2gtv8_g10_tiny_patch4_window7_769x769_80k_cityscapes_good.py"


# PRET="pretrained_models/swin_tiny_patch4_window7_224.pth"
# PRET="pretrained_models/swin_small_patch4_window7_224.pth"
PRET="pretrained_models/swin_base_patch4_window7_224.pth"
# PRET="pretrained_models/swin_base_patch4_window7_224_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window12_384_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window12_384.pth"


# RESUME="work_dirs/zz_upernet_swin_gtv8_g10_base_patch4_window7_769x769_160k_cityscapes/latest.pth"




srun python -u tools/train.py $CONFIG --options model.pretrained=$PRET --launcher="slurm" --seed 0 --deterministic ${@:3}
# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --resume-from=$RESUME --launcher="slurm" ${@:3} --seed 0 --deterministic ${@:3}
