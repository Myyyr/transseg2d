import torch
import os


def main():
	base_pth = "/users/t/themyrl/transseg2d/work_dirs/wacv_3009/"

	iter_48000 = "swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good/iter_48000.pth"
	iter_64000 = "swinunetv2gtv8nogmsa_g10_base_patch4_window7_512x512_160k_ade20k_good/iter_64000.pth"

	it48 = torch.load(os.path.join(base_pth, iter_48000))
	it64 = torch.load(os.path.join(base_pth, iter_64000))

	# print(type(it48))
	# print(it48.keys())
	# print(it48['state_dict'].keys())

	for k in list(it48['state_dict'].keys()):
		if 'global_token' in k:
			print(k)
			print(type(it48['state_dict'][k]))
			print(it48['state_dict'][k].shape)

			print("it48", it48['state_dict'][k].mean(), it48['state_dict'][k].std(), it48['state_dict'][k].min(), it48['state_dict'][k].max())
			print("it64", it64['state_dict'][k].mean(), it64['state_dict'][k].std(), it64['state_dict'][k].min(), it64['state_dict'][k].max())

			print("\n\n\n\n")





if __name__ == '__main__':
	main()