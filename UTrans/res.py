import json
import sys
import argparse
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model', type=str, required=True)
	parser.add_argument('-p','--path', type=str, default='/home/themyr_l/')
	parser.add_argument('-f','--fold', type=str, default='0')
	parser.add_argument('-h','--hd', dest="hd", action='store_true', default=False)
	# parser.set_defaults(feature=False)


	args = parser.parse_args()

	path = os.path.join(args.path, "nnUNetData/nnUNet_trained_models/nnUNet/3d_fullres_nnUNetPlansv2.1/Task017_BCV/")
	path = os.path.join(path, args.model, "fold_"+args.fold, "validation_raw_postprocessed/summary.json")

	with open(path, 'r') as f:
		data=json.load(f)['results']['mean']
		idxs = ""
		ress = ""
		for i in range(1,14):
			metric = "Dice"
			fact=100
			if args.hd:
				metric="Hausdorff Distance 95"
				fact=1
			res = round(float(data[str(i)][metric])*fact,2)
			res = str(res).replace('.',',')
			idx = str(i)
			idx = idx+" "*(len(res)-len(idx))

			ress += res+" "
			idxs += idx+" "
		print(args.model, 'fold_'+args.fold)
		print(idxs)
		print(ress)
		




if __name__ == '__main__':
	main()