#!/usr/bin/env python3
"""
Makes input dimensions (batch, height, width) dynamic.

usage:
# make batch + height/width all dynamic
python utils/static2dynamic.py --model modelname.onnx --hw

# keep batch static, only make H/W dynamic
python utils/static2dynamic.py --model modelname.onnx --no-batch --hw

"""

import argparse
import onnx
from onnx import shape_inference


def make_dynamic(model_path, output_path, dynamic_batch=True, dynamic_hw=False, 
                 batch_name='batch', height_name='height', width_name='width'):

    print(f'Loading model: {model_path}')
    model = onnx.load(model_path)
    
    # Get graph
    graph = model.graph
    
    # Print original input info
    print('\nOriginal inputs:')
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f'  {inp.name}: {shape}')
    
    # Modify input shapes
    for inp in graph.input:
        tensor_type = inp.type.tensor_type
        if tensor_type.HasField('shape'):
            dims = tensor_type.shape.dim
            
            # Typically NCHW format: [batch, channels, height, width]
            if len(dims) == 4:
                if dynamic_batch:
                    dims[0].dim_param = batch_name
                    dims[0].ClearField('dim_value')
                
                if dynamic_hw:
                    dims[2].dim_param = height_name
                    dims[2].ClearField('dim_value')
                    dims[3].dim_param = width_name
                    dims[3].ClearField('dim_value')
            
            # For 3D inputs [batch, seq, features] or similar
            elif len(dims) == 3:
                if dynamic_batch:
                    dims[0].dim_param = batch_name
                    dims[0].ClearField('dim_value')
    
    # Clear intermediate value info to allow shape inference to recompute
    while len(graph.value_info) > 0:
        graph.value_info.pop()
    
    # Modify output shapes to be dynamic (will be inferred)
    print('\nOriginal outputs:')
    for out in graph.output:
        shape = []
        if out.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f'  {out.name}: {shape}')
        
        # Clear output shapes - they'll be recomputed
        if out.type.tensor_type.HasField('shape'):
            dims = out.type.tensor_type.shape.dim
            if len(dims) >= 1 and dynamic_batch:
                dims[0].dim_param = batch_name
                dims[0].ClearField('dim_value')
            
            # For outputs that depend on H/W
            if dynamic_hw:
                for i, dim in enumerate(dims):
                    if i > 0:  # Skip batch
                        # Check if this dimension likely depends on spatial dims
                        val = dim.dim_value
                        if val and val > 1:
                            # Could be spatial-dependent, make it dynamic
                            pass  # Keep as is for now, shape inference will handle
    
    # Run shape inference
    print('\nRunning shape inference...')
    try:
        model = shape_inference.infer_shapes(model)
        print('Shape inference completed')
    except Exception as e:
        print(f'Warning: Shape inference failed: {e}')
        print('Continuing without full shape inference...')
    
    # Print new input/output info
    print('\nNew inputs:')
    for inp in model.graph.input:
        shape = []
        if inp.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f'  {inp.name}: {shape}')
    
    print('\nNew outputs:')
    for out in model.graph.output:
        shape = []
        if out.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f'  {out.name}: {shape}')
    
    # Save model
    print(f'\nSaving to: {output_path}')
    onnx.save(model, output_path)
    print('Done!')
    
    return model


def make_dynamic_with_ranges(model_path, output_path, input_shapes):
    
    print(f'Loading model: {model_path}')
    model = onnx.load(model_path)
    graph = model.graph
    
    # Print original input info
    print('\nOriginal inputs:')
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f'  {inp.name}: {shape}')
    
    # Modify specified inputs
    for inp in graph.input:
        if inp.name in input_shapes:
            new_shape = input_shapes[inp.name]
            dims = inp.type.tensor_type.shape.dim
            
            if len(dims) != len(new_shape):
                print(f'Warning: Shape mismatch for {inp.name}: {len(dims)} vs {len(new_shape)}')
                continue
            
            for i, (dim, new_dim) in enumerate(zip(dims, new_shape)):
                if isinstance(new_dim, str):
                    dim.dim_param = new_dim
                    dim.ClearField('dim_value')
                else:
                    dim.dim_value = new_dim
                    dim.ClearField('dim_param')
    
    # Clear intermediate value info
    while len(graph.value_info) > 0:
        graph.value_info.pop()
    
    # Run shape inference
    print('\nRunning shape inference...')
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f'Warning: Shape inference failed: {e}')
    
    # Print new info
    print('\nNew inputs:')
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f'  {inp.name}: {shape}')
    
    print('\nNew outputs:')
    for out in model.graph.output:
        shape = []
        if out.type.tensor_type.HasField('shape'):
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f'  {out.name}: {shape}')
    
    # Save
    print(f'\nSaving to: {output_path}')
    onnx.save(model, output_path)
    print('Done!')
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert static ONNX to dynamic ONNX')
    parser.add_argument('--model', '-m', type=str, required=True, help='Input static ONNX model path')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output dynamic ONNX model path')
    parser.add_argument('--batch', '-b', action='store_true', default=True, help='Make batch dimension dynamic (default: True)')
    parser.add_argument('--no-batch', action='store_true', help='Keep batch dimension static')
    parser.add_argument('--hw', action='store_true', help='Make height/width dimensions dynamic')
    parser.add_argument('--batch-name', type=str, default='batch', help='Name for batch dimension')
    parser.add_argument('--height-name', type=str, default='height', help='Name for height dimension')
    parser.add_argument('--width-name', type=str, default='width', help='Name for width dimension')
    
    args = parser.parse_args()
    
    # Generate output path if not specified
    if args.output is None:
        base = args.model.rsplit('.', 1)[0]
        args.output = f'{base}_dynamic.onnx'
    
    dynamic_batch = not args.no_batch
    
    make_dynamic(
        args.model, 
        args.output,
        dynamic_batch=dynamic_batch,
        dynamic_hw=args.hw,
        batch_name=args.batch_name,
        height_name=args.height_name,
        width_name=args.width_name
    )
