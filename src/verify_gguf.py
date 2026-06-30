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
            # Retrieve the Python representation of the field value
            val = field.parts[-1]
            # Convert bytes to string for readability if applicable
            if isinstance(val, (bytes, bytearray)):
                try:
                    val = val.decode('utf-8')
                except UnicodeDecodeError:
                    pass
            print(f"  {key}: {val} (type: {field.types})")
            
        print("\n--- Tensors ---")
        for tensor in reader.tensors:
            print(f"  Name: {tensor.name:<35} | Shape: {str(list(tensor.shape)):<15} | GGML Type: {tensor.tensor_type.name}")
            
        print("\nVerification read completed successfully!")
    except Exception as e:
        print(f"Error reading GGUF file: {e}")

if __name__ == "__main__":
    main()
