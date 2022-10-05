import torch
import os


def main():
	base_pth = "/users/t/themyrl/transseg2d/work_dirs/wacv_3009/"

	iter_48000 = "swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good/iter_48000.pth"
	iter_64000 = "swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good/iter_64000.pth"

	it48 = torch.load(os.path.join(base_pth, iter_48000))
	it64 = torch.load(os.path.join(base_pth, iter_64000))

	print(type(it48))





if __name__ == '__main__':
	main()