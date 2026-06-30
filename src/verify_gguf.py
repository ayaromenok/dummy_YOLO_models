import argparse
import gguf

def main():
    parser = argparse.ArgumentParser(description="Inspect GGUF file metadata and tensors.")
    parser.add_argument("file", help="Path to GGUF file")
    args = parser.parse_args()
    
    print(f"Reading GGUF file: {args.file}")
    try:
        reader = gguf.GGUFReader(args.file)
        
        print("\n--- Key-Value Fields / Metadata ---")
        for key, field in reader.fields.items():
            val = field.parts[-1]
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = val.decode('utf-8')
                except UnicodeDecodeError:
                    pass
            print(f"  {key}: {val} (type: {field.types})")
            
        print("\n--- Tensors (first 10) ---")
        for i, tensor in enumerate(reader.tensors):
            if i < 10:
                print(f"  Name: {tensor.name:<45} | Shape: {str(list(tensor.shape)):<15} | GGML Type: {tensor.tensor_type.name}")
        if len(reader.tensors) > 10:
            print(f"  ... and {len(reader.tensors) - 10} more tensors")
            
        print("\nVerification read completed successfully!")
    except Exception as e:
        print(f"Error reading GGUF file: {e}")

if __name__ == "__main__":
    main()
