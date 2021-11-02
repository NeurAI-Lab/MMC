#!/usr/bin/env python3

import argparse

import sdk.src.constants as const
from sdk.src.engine import Engine


def main():
    parser = argparse.ArgumentParser(description="Inference application.")
    parser.add_argument("--config-file", type=str, default=const.DEFAULT_CFG, metavar="FILE",
                        help="Path to config file (default: %(default)s).")
    parser.add_argument("--ckpt", type=str, default=const.DEFAULT_CKPT,
                        help="Trained weights (default: %(default)s).")
    parser.add_argument("--score_threshold", type=restricted_float, default=const.DEFAULT_SCORE_THRESHOLD,
                        help="Threshold value (default: %(default)s).")
    parser.add_argument("--dataset_type", type=str, choices=['coco', 'voc', 'had', 'ark', 'ark22', 'ark4', 'bdd', 'mgn'],
                        default=const.DEFAULT_DATASET_TYPE,
                        help='Specify dataset type (default: %(default)s).')
    parser.add_argument("--input", type=str, default=const.DEFAULT_INPUT_DIR,
                        help='Input directory/image to be predicted (default: %(default)s).')
    parser.add_argument("--output_dir", type=str, default=const.DEFAULT_OUTPUT_DIR,
                        help='Output directory that will contain the prediction(s) (default: %(default)s).')
    parser.add_argument("--output_format", type=str, choices=['img', 'json', 'json_nie', 'txt', 'xml'],
                        default=const.DEFAULT_OUTPUT_FORMAT,
                        help='Output format (default: %(default)s).')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose mode.')
    parser.add_argument("config_options", nargs=argparse.REMAINDER,
                        help="Configuration options that overwrites those from the configuration file.")
    args = parser.parse_args()
    if args.verbose:
        print(args)

    eng = Engine(config_file=args.config_file,
                 config_options=args.config_options,
                 ckpt=args.ckpt,
                 dataset_type=args.dataset_type,
                 score_threshold=args.score_threshold,
                 output_dir=args.output_dir,
                 output_format=args.output_format,
                 verbose=args.verbose)
    eng.load_model()
    eng.infer(args.input)


def restricted_float(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{val} not a floating-point literal')

    if val < 0.0 or val > 1.0:
        raise argparse.ArgumentTypeError(f'{val} not in range [0.0, 1.0]')
    return val


if __name__ == '__main__':
    main()
