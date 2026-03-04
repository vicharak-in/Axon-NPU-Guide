#!/usr/bin/env python3
"""
NanoDet ONNX -> RKNN Converter
"""

import os
import sys
import argparse

import onnx
from rknn.api import RKNN
from omegaconf import OmegaConf


# Config Loader
def load_nanodet_cfg(cfg_path):
    if not os.path.exists(cfg_path):
        print(f"ERROR: Config not found: {cfg_path}")
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)
    return cfg


def extract_nanodet_params(cfg):
    """
    Extract important inference params from NanoDet config
    """
    # Input size (W,H in YAML)
    w, h = cfg.data.val.input_size

    # Normalization
    mean_cfg, std_cfg = cfg.data.val.pipeline.normalize

    mean = list(mean_cfg)
    std  = list(std_cfg)

    # Head info (for postprocess later)
    head = cfg.model.arch.head

    num_classes = head.num_classes
    reg_max = head.reg_max
    strides = list(head.strides)

    return {
        "input_size": (h, w),   # convert to (H,W)
        "mean": mean,
        "std": std,
        "num_classes": num_classes,
        "reg_max": reg_max,
        "strides": strides,
    }

def convert_nanodet_to_rknn(
    onnx_path,
    cfg_path,
    output_path=None,
    target_platform="rk3588",
    do_quant=False,
    dataset_path=None,
    verbose=True,
):
    # Load YAML
    cfg = load_nanodet_cfg(cfg_path)
    params = extract_nanodet_params(cfg)

    H, W = params["input_size"]
    MEAN = params["mean"]
    STD = params["std"]

    # Output file
    if output_path is None:
        base = os.path.splitext(os.path.basename(onnx_path))[0]
        output_path = f"{base}_{H}x{W}.rknn"

    print("=" * 60)
    print("NanoDet ONNX -> RKNN Converter")
    print("=" * 60)

    print(f"ONNX Model:    {onnx_path}")
    print(f"Config:        {cfg_path}")
    print(f"Output:        {output_path}")
    print(f"Platform:      {target_platform}")
    print(f"Input Shape:   1x3x{H}x{W}")
    print(f"Mean:          {MEAN}")
    print(f"Std:           {STD}")
    print(f"Quantization:  {do_quant}")

    print("=" * 60)

    rknn = RKNN(verbose=verbose)
    print("--> Configuring RKNN...")
    if do_quant:
        if dataset_path is None:
            print("ERROR: Quantization requires --dataset")
            sys.exit(1)

        rknn.config(
            target_platform=target_platform,
            mean_values=[MEAN],
            std_values=[STD],
        )
    else:
        rknn.config(
            target_platform=target_platform,
            mean_values=[MEAN],
            std_values=[STD],
        )

    print("done")

    print("--> Loading ONNX model...")

    # Auto-detect input names and shapes from the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx_inputs = [inp.name for inp in onnx_model.graph.input]
    print(f"    ONNX input names: {onnx_inputs}")

    # Let RKNN auto-detect inputs; no need to specify inputs/input_size_list unless cropping the model graph.
    ret = rknn.load_onnx(model=onnx_path)

    if ret != 0:
        print("ERROR: Failed to load ONNX!")
        rknn.release()
        return ret

    print("done")

    print("--> Building RKNN...")

    if do_quant:
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset_path
        )
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print("ERROR: Build failed!")
        rknn.release()
        return ret

    print("done")

    # Export
    print("--> Exporting RKNN...")

    ret = rknn.export_rknn(output_path)

    if ret != 0:
        print("ERROR: Export failed!")
        rknn.release()
        return ret

    print("done")

    rknn.release()

    print("=" * 60)
    print("SUCCESS!")
    print(f"RKNN saved to: {output_path}")
    print("=" * 60)

    print("\nModel Info:")
    print(f"  Classes:  {params['num_classes']}")
    print(f"  RegMax:   {params['reg_max']}")
    print(f"  Strides:  {params['strides']}")

    print("\nNOTE:")
    print("Use BGR input (OpenCV) and feed 1x3xHxW tensor.")
    print("Postprocess must match NanoDetPlusHead.")

    return 0

def main():

    parser = argparse.ArgumentParser(
        description="NanoDet ONNX -> RKNN Converter"
    )

    parser.add_argument(
        "--onnx",
        required=True,
        help="Input ONNX model"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="NanoDet YAML config"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output RKNN file"
    )

    parser.add_argument(
        "--platform",
        default="rk3588",
        help="Target platform (default: rk3588)"
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization"
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset txt for quantization"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logs"
    )

    args = parser.parse_args()

    ret = convert_nanodet_to_rknn(
        onnx_path=args.onnx,
        cfg_path=args.config,
        output_path=args.output,
        target_platform=args.platform,
        do_quant=args.quantize,
        dataset_path=args.dataset,
        verbose=not args.quiet,
    )

    sys.exit(ret)


if __name__ == "__main__":
    main()
