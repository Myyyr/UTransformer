import json
import sys
import argparse
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model', type=str, required=True)
	parser.add_argument('-p','--path', type=str, default='~/')
	parser.add_argument('-f','--fold', type=str, default='0')

	args = parser.parse_args()

	path = os.path.join(args.path, "nnUNetData/nnUNet_trained_models/nnUNet/3d_fullres_nnUNetPlansv2.1/Task017_BCV/")
	path = os.path.join(path, args.model, "fold_"+args.fold, "validation_raw_postprocessed/summary.json")

	with open(path, 'r') as f:
		data=json.load(f)
		for key, value in data.items():
	    	print("Key:", key)




if __name__ == '__main__':
	main()