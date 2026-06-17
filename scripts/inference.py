#!/usr/bin/env python3
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudclear.inference.inference import CloudRemovalInference


def main():
    parser = argparse.ArgumentParser(description='CloudClear LISS-IV Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config YAML path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu or cuda)')
    parser.add_argument('--batch', action='store_true',
                        help='Process directory')
    args = parser.parse_args()

    import yaml

    def load_config(path):
        with open(path) as f:
            return yaml.safe_load(f)

    class AttrDict:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)

    config = AttrDict(load_config(args.config))

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    engine = CloudRemovalInference(args.checkpoint, config, device=device)

    if args.batch or os.path.isdir(args.input):
        output_dir = args.output or os.path.join(os.path.dirname(args.input), 'cleared')
        engine.process_directory(args.input, output_dir)
    else:
        output_path = args.output or args.input.rsplit('.', 1)[0] + '_clear.' + args.input.rsplit('.', 1)[1]
        engine.process_image(args.input, output_path)
        print(f"Result: {output_path}")


if __name__ == '__main__':
    main()
