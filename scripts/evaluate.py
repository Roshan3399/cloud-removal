#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudclear.utils.metrics import MetricsCalculator


def main():
    parser = argparse.ArgumentParser(description='CloudClear LISS-IV Evaluation')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory with predicted images')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory with ground truth images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output text file for results')
    parser.add_argument('--detailed', action='store_true',
                        help='Print per-sample metrics')
    args = parser.parse_args()

    from pathlib import Path
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    exts = ['*.tif', '*.tiff', '*.png', '*.jpg']
    pred_files = []
    for ext in exts:
        pred_files.extend(list(pred_dir.glob(ext)))

    if not pred_files:
        print(f"No files found in {args.pred_dir}")
        return

    metrics_calc = MetricsCalculator()
    all_metrics = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            gt_path = gt_dir / (pred_path.stem + '.png')
        if not gt_path.exists():
            print(f"  Skipping {pred_path.name}: no matching GT found")
            continue

        import cv2
        pred_img = cv2.imread(str(pred_path))
        gt_img = cv2.imread(str(gt_path))
        if pred_img is None or gt_img is None:
            print(f"  Skipping {pred_path.name}: failed to read")
            continue

        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        m = metrics_calc(pred_img, gt_img)
        all_metrics.append(m)

        if args.detailed:
            print(f"  [{pred_path.name}] PSNR={m['psnr']:.2f}  "
                  f"SSIM={m['ssim']:.4f}  SAM={m['sam']:.2f}°")

    if not all_metrics:
        print("No valid predictions evaluated.")
        return

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    result = "\n" + "=" * 50 + "\n"
    result += "  CloudClear LISS-IV — EVALUATION RESULTS\n"
    result += "=" * 50 + "\n"
    for k, v in avg.items():
        unit = "dB" if k == "psnr" else "°" if k == "sam" else ""
        result += f"  {k:>10}: {v:.4f} {unit}\n"
    result += "=" * 50 + "\n"
    result += f"  Samples: {len(all_metrics)}\n"
    result += "=" * 50

    print(result)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)


if __name__ == '__main__':
    main()
