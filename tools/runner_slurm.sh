#!/bin/bash
#SBATCH --job-name=sunnndclip1     # job name
#SBATCH --ntasks=8                  # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=99:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/sunnndclip1.out # output file name # add %j to id the job
#SBATCH --error=logs/sunnndclip1.err  # error file name # add %j to id the job
#SBATCH -C v100-32g

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2
module load python/3.7.10


# RESUME="work_dirs/swinunet_tiny_patch4_window7_512x512_160k_ade20k/iter_128000.pth"
# RESUME="work_dirs/swinunetgtv1_tiny_patch4_window7_512x512_160k_ade20k/iter_32000.pth"
# RESUME="work_dirs/swinunetgtv2_tiny_patch4_window7_512x512_160k_ade20k/iter_32000.pth"
# RESUME="work_dirs/swinunetv2gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k/latest.pth"
# RESUME="work_dirs/swinunetv2gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k/latest.pth"
# RESUME="work_dirs/swinunetv2gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k/latest.pth"
# RESUME="work_dirs/swinunetv2gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k_good/latest.pth"
# RESUME="work_dirs/swinunetv2gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k_good/latest.pth"



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
# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_base_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_tiny_patch4_window7_769x769_160k_cityscapes.py"
# CONFIG="configs/swinunetv2gtv3/swinunetv2gtv3_g5_tiny_patch4_window7_769x769_160k_cityscapes.py"

# CONFIG="configs/swinunetv2/swinunetv2_small_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2/swinunetv2_base_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2/swinunetv2_tiny_patch4_window7_769x769_160k_cityscapes.py"

# CONFIG="configs/swinunetv2/swinunetv2_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv7/swinunetv2gtv7_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv7/swinunetv2gtv7_g5_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv7/swinunetv2gtv7_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"

# CONFIG="configs/swinunetv2crossattentionupsample/swinunetv2_cross_attention_upsample_tiny_patch4_window7_512x512_160k_ade20k.py"

# --------------------------------------------- 

# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k.py"

# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g5_small_patch4_window7_512x512_160k_ade20k.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g5_base_patch4_window7_512x512_160k_ade20k.py"


# CONFIG="configs/swinupergtv8/upernet_swin_gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k.py"

# --------------------------------------------- 
# CONFIG="configs/swinunetv2gtv7/swinunetv2gtv7_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv7/swinunetv2gtv7_g5_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv7/swinunetv2gtv7_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# --------------------------------------------- 
## CONFIG="configs/swinunetv2/swinunetv2_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2/swinunetv2_small_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2/swinunetv2_base_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2/swinunetv2_tiny_patch4_window7_769x769_160k_cityscapes_good.py"

## CONFIG="configs/orininal_swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_good.py"

## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"

## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g1_small_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g1_base_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g1_tiny_patch4_window7_769x769_160k_cityscapes_good.py"
## CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g10_small_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g10_base_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/swinunetv2gtv8_g10_tiny_patch4_window7_769x769_160k_cityscapes_good.py"

## CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2gtv5/swinunetv2gtv5_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"

## CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g5_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinunetv2gtv4/swinunetv2gtv4_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"

## CONFIG="configs/swinupergtv8/upernet_swin_gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/swinupergtv8/upernet_swin_gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"


## CONFIG="configs/orininal_no_swin/upernet_no_swin_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/noswinupergtv8/upernet_no_swin_gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
## CONFIG="configs/noswinupergtv8/upernet_no_swin_gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"

# ..............................................
# CONFIG="configs/swinunetv2/zswinunetv2_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2/zswinunetv2_small_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2/zswinunetv2_base_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2/zswinunetv2_tiny_patch4_window7_769x769_160k_cityscapes_good.py"

# CONFIG="configs/orininal_swin/zupernet_swin_tiny_patch4_window7_512x512_160k_ade20k_good.py"

# CONFIG="configs/swinunetv2gtv8/zswinunetv2gtv8_g1_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/zswinunetv2gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# CONFIG="configs/swinunetv2gtv8/zswinunetv2gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"
# ..............................................
# CONFIG="configs/orininal_swin/zz_upernet_swin_base_patch4_window7_769x769_160k_cityscapes.py"
# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_base_patch4_window7_769x769_160k_cityscapes.py"

# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_base_patch4_window7_512x512_160k_ade20k.py"

# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g15_tiny_patch4_window7_512x512_160k_ade20k.py" #X
# # CONFIG="configs/swinunetv2gtv8/zz_swinunetv2gtv8_g15_tiny_patch4_window7_512x512_160k_ade20k.py" #X

# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g5_tiny_patch4_window7_512x512_160k_ade20k.py" #X

# ..............................................
# CONFIG="configs/swinunetv2/zz_swinunetv2_tiny_patch4_window7_769x769_80k_cityscapes_good.py"
# CONFIG="configs/swinunetv2/zz_swinunetv2_tiny_patch4_window7_769x769_160k_cityscapes_good.py" #X
# CONFIG="configs/swinunetv2gtv8/zz_swinunetv2gtv8_g10_tiny_patch4_window7_769x769_160k_cityscapes_good.py"
# CONFIG="configs/swinunetv2gtv8/zz_swinunetv2gtv8_g10_tiny_patch4_window7_769x769_80k_cityscapes_good.py"
# ..............................................
# CONFIG="configs/swinunetv2gtv8/zzz_swinunetv2gtv8_g10_base_patch4_window7_769x769_160k_cityscapes_good.py"
# CONFIG="configs/swinunetv2/zzz_swinunetv2_base_patch4_window7_769x769_160k_cityscapes_good.py" 

# ..............................................

# CONFIG="configs/segformergtgamma/segformer.gt.gamma.g10.b4.512x512.ade.160k.py"
# CONFIG="configs/segformergtgamma/segformer.gt.gamma.g1.b4.512x512.ade.160k.py"
# CONFIG="configs/segformergtgamma/segformer.gt.gamma.g5.b4.512x512.ade.160k.py"

# ...............................................

# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_base_patch4_window7_769x769_160k_cityscapes.py"

# CONFIG="configs/swinunetv2gtv12/swinunetv2gtv12.g10.tiny.patch4.window7.512x512.160k.ade20k.jz.py"
# CONFIG="configs/swinunetv2gtv13/swinunetv2gtv13.g10.tiny.patch4.window7.512x512.160k.ade20k.jz.py"




# CONFIG="configs/swinunetv2gtv9/swinunetv2gtv9.g1.tiny.patch4.window7.512x512.160k.ade20k.jz.py"


# ...............................................
# V14



# CONFIG="configs/swinunetv2gtv14/swinunetv2gtv14.g10.tiny.patch4.window7.512x512.160k.ade20k.jz.py"
# CONFIG="configs/swinunetv2gtv14/swinunetv2gtv14.g10.base.patch4.window7.512x512.160k.ade20k.jz.py"
# CONFIG="configs/swinupergtv14/upernet.swin.gtv14.g10.base.patch4.window7.512x512.160k.ade20k.py"
# CONFIG="configs/swinupergtv14/upernet.swin.gtv14.g1.tiny.patch4.window7.512x512.160k.ade20k.py" #dbg
# CONFIG="configs/swinunetv2gtv14/swinunetv2gtv14.g1.tiny.patch4.window7.512x512.160k.ade20k.jz.py"
# ...............................................
# SEGFORMER


# CONFIG="configs/segformer/segformer.b5.769x769.city.160k.py" # sfb5citc
# RESUME="work_dirs/segformer.b5.769x769.city.160k/latest.pth"
# CONFIG="configs/segformer/segformer.b4.512x512.ade.160k.py"

# CONFIG="configs/segformer/segformer.b5.769x769.city.160k.bs8.py" # sfb5citb8
# RESUME="work_dirs/segformer.b5.769x769.city.160k.bs8/latest.pth"


# ...............................................
# v8
# CONFIG="configs/swinupergtv8/zz.upernet.swin.gtv8.g10.base.patch4.window7.1024x1024.160k.cityscapes.py" # zgup1024cit # city1024
# CONFIG="configs/swinupergtv8/zz.upernet.swin.gtv8.g10.base.patch4.window7.1024x1024.160k.cityscapes.bs2.py" # zgup1024cit # city1024bs2
# CONFIG="configs/swinupergtv8/upernet_swin_gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"

# RESUME="work_dirs/upernet_swin_gtv8_g10_tiny_patch4_window7_512x512_160k_ade20k_good/latest.pth"






# -------------------------------------------------------------------------


# CONFIG="configs/orininal_swin/zz.upernet.swin.tiny.patch4.window7.769x769.160k.cityscapes.py"
# CONFIG="configs/swinupergtv8/zz.upernet.swin.gtv8.g10.tiny.patch4.window7.769x769.160k.cityscapes.py" #X
# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_small_patch4_window7_512x512_160k_ade20k.py"


# CONFIG="configs/swinunetv2gtv8/zzz_swinunetv2gtv8_g10_base_patch4_window7_769x769_160k_cityscapes_good.py"
# CONFIG="configs/swinunetv2/zzz_swinunetv2_base_patch4_window7_769x769_160k_cityscapes_good.py" 


# -------------------------------------------------------------------------
# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_base_patch4_window7_512x512_160k_ade20k.py" #glmbupg10


# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g10_small_patch4_window7_512x512_160k_ade20k.py" # gspg10
# CONFIG="configs/swinupergtv8/zz_upernet_swin_gtv8_g15_tiny_patch4_window7_512x512_160k_ade20k.py" # gtpg15
# CONFIG="configs/swinunetv2gtv8/zz_swinunetv2gtv8_g15_tiny_patch4_window7_512x512_160k_ade20k.py" # gtng15



#--------------------------------------------------------------------------
# CONFIG="configs/swinunetv2gtv8nogmsa/zswinunetv2gtv8nogmsa_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"	    #suntngg10
# CONFIG="configs/swinunetv2gtv8nogmsa/swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good.py"	#sunbngg10

# CONFIG="configs/swinupergtv8nogmsa/upernet_swin_gtv8_nogmsa_g10_tiny_patch4_window7_512x512_160k_ade20k_good.py"	#suptngg10
# CONFIG="configs/swinupergtv8nogmsa/zz_upernet_swin_gtv8_nogmsa_g10_base_patch4_window7_512x512_160k_ade20k.py"	#supbngg10


#--------------------------------------------------------------------------
# CONFIG="configs/swinupergtv8nogmsa/zz_upernet_swin_gtv8_nogmsa_g10_base_patch4_window7_512x512_240k_ade20k.py"	#supn24
# CONFIG="configs/swinupergtv8nogmsa/zz_upernet_swin_gtv8_nogmsa_g10_base_patch4_window7_512x512_320k_ade20k.py"	#supn32
# CONFIG="configs/swinunetv2gtv8nogmsa/swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good.py"	#sunnnd
# CONFIG="configs/swinunetv2gtv8nogmsa/swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good_debug.py"	#sunnnddbg
# CONFIG="configs/swinunetv2gtv8nogmsa/swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good_lr.py"	#sunnndlr
# CONFIG="configs/swinunetv2gtv8nogmsa/swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good_lrh.py"	#sunnndlrh
# CONFIG="configs/swinunetv2gtv8nogmsaclip/swinunetv2gtv8nogmsaclip20_g10_base_patch4_window7_512x512_160k_ade20k_good.py"	#sunnndclip
CONFIG="configs/swinunetv2gtv8nogmsaclip/swinunetv2gtv8nogmsaclip1_g10_base_patch4_window7_512x512_160k_ade20k_good.py"	#sunnndclip1




# PRET="pretrained_models/swin_tiny_patch4_window7_224.pth"
# PRET="pretrained_models/swin_small_patch4_window7_224.pth"
PRET="pretrained_models/swin_base_patch4_window7_224.pth"
# PRET="pretrained_models/swin_tiny_patch4_window7_224_22k.pth"
# PRET="pretrained_models/swin_base_patch4_window7_224_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window7_224_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window12_384_22k.pth"
# PRET="pretrained_models/swin_tiny_patch4_window7_224_22k.pth"
# PRET="pretrained_models/swin_base_patch4_window7_224_22k.pth"
# PRET="pretrained_models/swin_small_patch4_window7_224_22k.pth"
## PRET="pretrained_models/swin_base_patch4_window12_384.pth"

# PRET="pretrained_models/glam_swin_tiny_patch4_window7_224.pth"
# PRET="pretrained_models/glam_swin_base_patch4_window7_224.pth"


# RESUME="work_dirs/zz_upernet_swin_gtv8_g10_base_patch4_window7_769x769_160k_cityscapes/latest.pth"
# RESUME="work_dirs/swinunetv2gtv9.g1.tiny.patch4.window7.512x512.160k.ade20k.jz/latest.pth"
# RESUME="work_dirs/swinunetv2gtv12.g10.tiny.patch4.window7.512x512.160k.ade20k.jz/latest.pth"



# swin
srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --options model.pretrained=$PRET --launcher="slurm" --seed 0 --deterministic --no-validate ${@:3}
# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --options model.pretrained=$PRET --launcher="slurm" --no-validate ${@:3}

# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --resume-from=$RESUME --launcher="slurm" ${@:3} --seed 0 --deterministic --no-validate ${@:3} 
# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/test.py $CONFIG $RESUME --launcher="slurm" --eval mIoU ${@:3}




#segformer
# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --launcher="slurm" --seed 0 --deterministic ${@:3} #segformer
# srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/train.py $CONFIG --resume-from=$RESUME --launcher="slurm" --seed 0 --deterministic ${@:3} #segformer


#