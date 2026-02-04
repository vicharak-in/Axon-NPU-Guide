#!/usr/bin/env python3
"""
Usage:
    python onnx2rknn.py <model.onnx> [output.rknn] --graphsz 320,320 480,480 640,360

Note: ONNX model opset must be <= 19 for RKNN conversion.
      export example: yolo export model=yolov8n.pt format=onnx opset=12
      use a different venv for rknn tasks***
"""

from rknn.api import RKNN
import sys
import os
import argparse


# default shapes for YOLOv8. these are the resolutions the model will support at runtime
DEFAULT_YOLO_SHAPES = [
    [[1, 3, 320, 320]],
    [[1, 3, 480, 480]],
    [[1, 3, 640, 640]],
    [[1, 3, 1792, 1792]],
]


def parse_graphsz(graphsz_list):
    """
    Parse list of 'H,W' strings into RKNN input_shapes format.
    """
    shapes = []
    for sz in graphsz_list:
        try:
            h, w = map(int, sz.split(','))
            shapes.append([[1, 3, h, w]])
        except ValueError:
            print(f"ERROR: Invalid shape format '{sz}'. Expected H,W (e.g., 640,480)")
            sys.exit(1)
    return shapes


def convert_onnx_to_rknn(
    onnx_path: str,
    rknn_path: str = None,
    target_platform: str = 'rk3588',
    input_shapes: list = None,
    do_quantization: bool = False,
    dataset_path: str = None,
    verbose: bool = True
):
    
    # RKNN will compile the model for each of these shapes
    if input_shapes is None:
        input_shapes = DEFAULT_YOLO_SHAPES
    
    # Generate output path if not provided
    if rknn_path is None:
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        # Create shape string: 320x320-640x360-1280x720
        shapes_str = '-'.join([f"{s[0][2]}x{s[0][3]}" for s in input_shapes])
        rknn_path = f'{base_name}_{shapes_str}.rknn'
    
    print(f'=' * 60)
    print(f'ONNX to RKNN Converter')
    print(f'=' * 60)
    print(f'Input ONNX:      {onnx_path}')
    print(f'Output RKNN:     {rknn_path}')
    print(f'Target Platform: {target_platform}')
    print(f'Quantization:    {do_quantization}')
    print(f'Dynamic Shapes:  {len(input_shapes)} configurations')
    for i, shape in enumerate(input_shapes):
        print(f'  Shape {i+1}: {shape[0]} -> {shape[0][2]}x{shape[0][3]}')
    print(f'=' * 60)
    
    # Create RKNN object
    rknn = RKNN(verbose=verbose)
    
    # Configure RKNN
    print('--> Configuring RKNN...')
    
    if do_quantization:
        rknn.config(
            target_platform=target_platform,
            dynamic_input=input_shapes,
            mean_values=[[0, 0, 0]] * len(input_shapes),
            std_values=[[255, 255, 255]] * len(input_shapes),
        )
    else:
        rknn.config(
            target_platform=target_platform,
            dynamic_input=input_shapes,
        )
    print('done')
    
    # Load ONNX model
    print('--> Loading ONNX model...')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f'ERROR: Failed to load ONNX model! (code: {ret})')
        rknn.release()
        return ret
    print('done')
    
    # Build RKNN model
    print('--> Building RKNN model...')
    if do_quantization:
        if dataset_path is None:
            print('ERROR: Quantization requires a dataset file!')
            rknn.release()
            return -1
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
    else:
        ret = rknn.build(do_quantization=False)
    
    if ret != 0:
        print(f'ERROR: Failed to build RKNN model! (code: {ret})')
        rknn.release()
        return ret
    print('done')
    
    # Export RKNN model
    print('--> Exporting RKNN model...')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f'ERROR: Failed to export RKNN model! (code: {ret})')
        rknn.release()
        return ret
    print('done')
    
    # Cleanup
    rknn.release()
    
    print(f'=' * 60)
    print(f'SUCCESS: RKNN model exported to: {rknn_path}')
    print(f'=' * 60)
    print()
    print('USAGE NOTE:')
    print('At runtime, input tensor shape must match one of these:')
    for shape in input_shapes:
        h, w = shape[0][2], shape[0][3]
        print(f'  --size {h},{w}')
    print()
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to RKNN format with multi-shape support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Square shapes (traditional)
  python onnx2rknn.py model.onnx --graphsz 320,320 480,480 640,640

  # Rectangular shapes (16:9 video)
  python onnx2rknn.py model.onnx --graphsz 640,360 1280,720

  # Mixed shapes
  python onnx2rknn.py model.onnx --graphsz 320,320 640,480 1280,720

  # With quantization
  python onnx2rknn.py model.onnx --graphsz 640,640 --quantize --dataset dataset.txt

  # Custom output name
  python onnx2rknn.py model.onnx output.rknn --graphsz 640,640
        """
    )
    
    parser.add_argument('onnx', help='Input ONNX model path')
    parser.add_argument('rknn', nargs='?', default=None, 
                        help='Output RKNN model path (auto-generated if not specified)')
    parser.add_argument('--graphsz', nargs='+', metavar='H,W',
                        help='Graph sizes as H,W pairs (e.g., --graphsz 320,320 640,480 1280,720)')
    parser.add_argument('--platform', default='rk3588',
                        help='Target platform (default: rk3588)')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable quantization (requires --dataset)')
    parser.add_argument('--dataset', default=None,
                        help='Dataset file for quantization (txt with image paths)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbose output')
    
    args = parser.parse_args()
    
    # Parse input shapes
    if args.graphsz:
        input_shapes = parse_graphsz(args.graphsz)
    else:
        print('No --graphsz specified, using defaults: 320x320, 480x480, 640x640, 1792x1792')
        input_shapes = DEFAULT_YOLO_SHAPES
    
    ret = convert_onnx_to_rknn(
        onnx_path=args.onnx,
        rknn_path=args.rknn,
        target_platform=args.platform,
        input_shapes=input_shapes,
        do_quantization=args.quantize,
        dataset_path=args.dataset,
        verbose=not args.quiet
    )
    
    sys.exit(ret)