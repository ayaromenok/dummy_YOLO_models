import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gguf
from ptflops import get_model_complexity_info
from ultralytics.nn.modules import Conv, C3k2, C2PSA, SPPF

class DummyYOLO(nn.Module):
    def __init__(self, first_conv_dim: int, num_conv_layers: int):
        super().__init__()
        
        # Stem layer: Downsamples input spatial shape by factor of 2
        layers = [Conv(3, first_conv_dim, 3, 2, 1)]
        
        # Dynamic backbone stages consisting of downsampling Conv and C3k2 attention blocks
        for _ in range(1, num_conv_layers):
            layers.append(Conv(first_conv_dim, first_conv_dim, 3, 2, 1))
            layers.append(C3k2(first_conv_dim, first_conv_dim, n=2, c3k=True, attn=True))
            
        self.backbone = nn.Sequential(*layers)
        
        # C2PSA requires the channel size (c1 * expansion_ratio) to be at least 64,
        # which means input channels must be >= 128 (with default expansion 0.5) to avoid DivisionByZero.
        # We apply a dynamic channel projection workaround for dimensions < 128.
        self.c_mid = max(128, first_conv_dim)
        if first_conv_dim < 128:
            self.project_up = Conv(first_conv_dim, self.c_mid, 1)
            self.project_down = Conv(self.c_mid, first_conv_dim, 1)
        else:
            self.project_up = nn.Identity()
            self.project_down = nn.Identity()
            
        self.sppf = SPPF(self.c_mid, self.c_mid, 5)
        self.c2psa = C2PSA(self.c_mid, self.c_mid, 1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(first_conv_dim, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.project_up(x)
        x = self.sppf(x)
        x = self.c2psa(x)
        x = self.project_down(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
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
        data = (tensor.detach().cpu() * 1000).to(torch.int32).numpy()
        return data, gguf.GGMLQuantizationType.I32
    elif data_format == "int16":
        data = (tensor.detach().cpu() * 1000).to(torch.int16).numpy()
        return data, gguf.GGMLQuantizationType.I16
    elif data_format == "int8":
        data = (tensor.detach().cpu() * 100).to(torch.int8).numpy()
        return data, gguf.GGMLQuantizationType.I8
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

def main():
    parser = argparse.ArgumentParser(description="Create a configurable dummy YOLO26 model and save in GGUF/PyTorch formats.")
    parser.add_argument("--first-conv-dim", type=int, default=16, help="Output channels of the first Conv2d layer (default: 16)")
    parser.add_argument("--num-conv-layers", type=int, default=3, help="Total number of backbone stages (default: 3)")
    parser.add_argument("--data-format", type=str, default="float32",
                        choices=["float32", "float16", "float8", "int32", "int16", "int8"],
                        help="Target data format (default: float32)")
    parser.add_argument("--output-gguf", type=str, default="dummy_yolo26.gguf", help="Output GGUF file path (default: dummy_yolo26.gguf)")
    parser.add_argument("--output-pt", type=str, default="dummy_yolo26.pt", help="Output PyTorch checkpoint file path (default: dummy_yolo26.pt)")
    
    args = parser.parse_args()
    
    print(f"Creating Dummy YOLO26 model (first_conv_dim={args.first_conv_dim}, num_conv_layers={args.num_conv_layers})...")
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
    print(f"Initializing GGUF writer for architecture 'yolo26'...")
    writer = gguf.GGUFWriter(args.output_gguf, "yolo26")
    
    writer.add_name("YOLO26 Dummy Model")
    writer.add_description(f"Configurable dummy YOLO26 model with {args.num_conv_layers} backbone stages (first layer dim={args.first_conv_dim}) in {args.data_format} format")
    
    # Add hyperparameters/metadata
    writer.add_uint32("yolo26.first_conv_dim", args.first_conv_dim)
    writer.add_uint32("yolo26.num_conv_layers", args.num_conv_layers)
    writer.add_string("yolo26.data_format", args.data_format)
    
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
