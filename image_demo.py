# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import json

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('img', help='Image file')
	parser.add_argument('config', help='Config file')
	parser.add_argument('checkpoint', help='Checkpoint file')
	parser.add_argument('--out-file', default=None, help='Path to output file')
	parser.add_argument(
    	'--device', default='cpu', help='Device used for inference')
	parser.add_argument(
    	'--palette',
    	default='dota',
    	choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
    	help='Color palette used for visualization')
	parser.add_argument(
    	'--score-thr', type=float, default=0.3, help='bbox score threshold')
	args = parser.parse_args()
	return args


def main(args):
    try:
        # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint, device=args.device)
        # test a single image
        result = inference_detector(model, args.img)
        # show the results
        show_result_pyplot(
            model,
            args.img,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=args.out_file)
        
        # Convert result to JSON serializable format
        result_json = []
        for class_results in result:
            if len(class_results) > 0:
                result_json.append(class_results.tolist())
            else:
                result_json.append([])

        # Return the result for further processing
        return result_json
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return []

if __name__ == '__main__':
	args = parse_args()
	result = main(args)
	# Print the result to stdout for capturing in the calling script
	print(json.dumps(result))
