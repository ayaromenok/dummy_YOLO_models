#python3 generate_dummy_yolo.py \
#  --first-conv-dim 32 \
#  --num-conv-layers 4 \
#  --data-format float16 \
#  --output-gguf dummy_yolo_F16.gguf \
#  --output-pt dummy_yolo_F16.pt
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gguf
from ptflops import get_model_complexity_info

class DummyYOLO(nn.Module):
    def __init__(self, first_conv_dim: int, num_conv_layers: int):
        super().__init__()
        
        # Build Conv2d layers sequence
        layers = []
        in_channels = 3
        for i in range(num_conv_layers):
            out_channels = first_conv_dim
            # Standard Conv2d + SiLU activation (commonly used in YOLO)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.SiLU())
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(first_conv_dim, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

def to_gguf_tensor(tensor: torch.Tensor, data_format: str):
    if data_format == "float32":
        data = tensor.detach().cpu().to(torch.float32).numpy()
        return data, gguf.GGMLQuantizationType.F32
    elif data_format == "float16":
        data = tensor.detach().cpu().to(torch.float16).numpy()
        return data, gguf.GGMLQuantizationType.F16
    elif data_format == "float8":
        # Cast to float8_e4m3fn in PyTorch
        t_f8 = tensor.detach().cpu().to(torch.float8_e4m3fn)
        # View as raw uint8 bytes
        t_raw = t_f8.view(torch.uint8)
        # Convert to numpy and view as int8 for GGUFWriter compatibility
        data = t_raw.numpy().view(np.int8)
        return data, gguf.GGMLQuantizationType.I8
    elif data_format == "int32":
        # Scale to avoid zero-out
        data = (tensor.detach().cpu() * 1000).to(torch.int32).numpy()
        return data, gguf.GGMLQuantizationType.I32
    elif data_format == "int16":
        # Scale to avoid zero-out
        data = (tensor.detach().cpu() * 1000).to(torch.int16).numpy()
        return data, gguf.GGMLQuantizationType.I16
    elif data_format == "int8":
        # Scale to avoid zero-out
        data = (tensor.detach().cpu() * 100).to(torch.int8).numpy()
        return data, gguf.GGMLQuantizationType.I8
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

def main():
    parser = argparse.ArgumentParser(description="Create a configurable dummy YOLO model and save in GGUF/PyTorch formats.")
    parser.add_argument("--first-conv-dim", type=int, default=16, help="Output channels of the first Conv2d layer (default: 16)")
    parser.add_argument("--num-conv-layers", type=int, default=3, help="Total number of Conv2d layers (default: 3)")
    parser.add_argument("--data-format", type=str, default="float32",
                        choices=["float32", "float16", "float8", "int32", "int16", "int8"],
                        help="Target data format (default: float32)")
    parser.add_argument("--output-gguf", type=str, default="dummy_yolo.gguf", help="Output GGUF file path (default: dummy_yolo.gguf)")
    parser.add_argument("--output-pt", type=str, default="dummy_yolo.pt", help="Output PyTorch checkpoint file path (default: dummy_yolo.pt)")
    
    args = parser.parse_args()
    
    print(f"Creating Dummy YOLO model (first_conv_dim={args.first_conv_dim}, num_conv_layers={args.num_conv_layers})...")
    model = DummyYOLO(args.first_conv_dim, args.num_conv_layers)
    
    # Calculate computational complexity using ptflops (run in FP32 on CPU)
    input_shape = (3, 640, 640)
    print(f"Estimating model complexity with input shape {input_shape}...")
    try:
        macs, params = get_model_complexity_info(
            model, 
            input_shape, 
            as_strings=True, 
            print_per_layer_stat=False,
            verbose=False
        )
        print(f"Computational Complexity (MACs): {macs}")
        print(f"Number of Parameters: {params}")
    except Exception as e:
        print(f"Warning: ptflops estimation failed: {e}")
    
    # Cast weights for PyTorch checkpoint
    print(f"Converting PyTorch checkpoint to format '{args.data_format}'...")
    pt_state_dict = {}
    for name, tensor in model.state_dict().items():
        if args.data_format == "float32":
            pt_state_dict[name] = tensor.to(torch.float32)
        elif args.data_format == "float16":
            pt_state_dict[name] = tensor.to(torch.float16)
        elif args.data_format == "float8":
            pt_state_dict[name] = tensor.to(torch.float8_e4m3fn)
        elif args.data_format == "int32":
            pt_state_dict[name] = (tensor * 1000).to(torch.int32)
        elif args.data_format == "int16":
            pt_state_dict[name] = (tensor * 1000).to(torch.int16)
        elif args.data_format == "int8":
            pt_state_dict[name] = (tensor * 100).to(torch.int8)
            
    checkpoint = {
        'epoch': 0,
        'model': pt_state_dict,
        'optimizer': None
    }
    
    print(f"Saving PyTorch checkpoint to {args.output_pt}...")
    torch.save(checkpoint, args.output_pt)
    print("PyTorch checkpoint save successful!")
    
    # Initialize GGUF Writer and write tensors
    print(f"Initializing GGUF writer for architecture 'yolo'...")
    writer = gguf.GGUFWriter(args.output_gguf, "yolo")
    
    writer.add_name("YOLO Dummy Model")
    writer.add_description(f"Configurable dummy YOLO model with {args.num_conv_layers} Conv2d layers (first layer dim={args.first_conv_dim}) in {args.data_format} format")
    
    # Add hyperparameters/metadata
    writer.add_uint32("yolo.first_conv_dim", args.first_conv_dim)
    writer.add_uint32("yolo.num_conv_layers", args.num_conv_layers)
    writer.add_string("yolo.data_format", args.data_format)
    
    print("Writing tensors to GGUF...")
    for name, tensor in model.state_dict().items():
        data, dtype = to_gguf_tensor(tensor, args.data_format)
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
            
        writer.add_tensor(name, data, raw_dtype=dtype)
        print(f"  Added tensor: {name} | shape: {data.shape} | dtype: {data.dtype} | GGUF raw_dtype: {dtype.name}")
        
    print(f"Saving GGUF file to {args.output_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("GGUF save successful!")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
