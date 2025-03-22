import scipy.io
import pandas as pd
import numpy as np
import os
import h5py

def convert_structured_mat(mat_file):
    # Load the MAT file
    mat_data = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    
    # Get base filename
    base_filename = os.path.splitext(os.path.basename(mat_file))[0]
    
    # Process each variable
    for key in mat_data.keys():
        if key.startswith('__'):  # Skip metadata
            continue
            
        print(f"Processing variable: {key}")
        
        # Get the data
        data = mat_data[key]
        
        # Check if it's a MATLAB struct
        if hasattr(data, '_fieldnames'):
            print(f"Found MATLAB struct with fields: {data._fieldnames}")
            
            # Create a directory for this battery
            output_dir = f"processed_{base_filename}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract each field as a separate CSV
            for field in data._fieldnames:
                if hasattr(data, field):
                    field_data = getattr(data, field)
                    
                    # Handle different data types
                    if isinstance(field_data, np.ndarray):
                        if field_data.ndim <= 2:  # 1D or 2D array
                            df = pd.DataFrame(field_data)
                            df.to_csv(f"{output_dir}/{field}.csv", index=False)
                            print(f"  Saved {field}.csv")
                        else:
                            # Handle multi-dimensional arrays
                            for i in range(field_data.shape[0]):
                                df = pd.DataFrame(field_data[i])
                                df.to_csv(f"{output_dir}/{field}_{i}.csv", index=False)
                            print(f"  Saved {field}_*.csv ({field_data.shape[0]} files)")
                    elif hasattr(field_data, '_fieldnames'):
                        # Nested struct
                        print(f"  Found nested struct in {field}")
                        nested_dir = f"{output_dir}/{field}"
                        os.makedirs(nested_dir, exist_ok=True)
                        
                        for nested_field in field_data._fieldnames:
                            if hasattr(field_data, nested_field):
                                nested_data = getattr(field_data, nested_field)
                                if isinstance(nested_data, np.ndarray) and nested_data.ndim <= 2:
                                    df = pd.DataFrame(nested_data)
                                    df.to_csv(f"{nested_dir}/{nested_field}.csv", index=False)
                                    print(f"    Saved {field}/{nested_field}.csv")
                    else:
                        print(f"  Skipping {field}: not an array or struct")
            
            print(f"Processed {key} into directory: {output_dir}")
        else:
            print(f"Skipping {key}: not a MATLAB struct")
            
            
def process_array(array, output_base):
    """Process a NumPy array and convert to CSV if possible"""
    print(f"Type: {type(array)}")
    
    if isinstance(array, np.ndarray):
        print(f"Shape: {array.shape}")
        
        # Handle different array types
        if len(array.shape) == 2:
            # Check if it's a structured array
            if array.dtype.names is not None:
                print("Detected structured array")
                # Convert structured array to DataFrame
                df = pd.DataFrame({name: array[name] for name in array.dtype.names})
                df.to_csv(f"{output_base}.csv", index=False)
                print(f"Saved structured array to {output_base}.csv")
            else:
                # Simple 2D array
                try:
                    df = pd.DataFrame(array)
                    df.to_csv(f"{output_base}.csv", index=False)
                    print(f"Saved 2D array to {output_base}.csv")
                except ValueError:
                    # If direct conversion fails, try different approach
                    print("Direct conversion failed, trying alternative method...")
                    save_complex_array(array, output_base)
        
        # For objects that might contain nested structures (e.g., MATLAB structs)
        elif array.shape == (1, 1) and array.dtype == np.dtype('O'):
            nested_obj = array[0, 0]
            print(f"Detected possible nested structure: {type(nested_obj)}")
            
            if isinstance(nested_obj, dict):
                process_dict(nested_obj, output_base)
            elif isinstance(nested_obj, np.ndarray):
                process_array(nested_obj, f"{output_base}_nested")
            else:
                print(f"Unknown nested type: {type(nested_obj)}")
        
        # Multi-dimensional arrays
        elif len(array.shape) > 2:
            print(f"Processing {len(array.shape)}-dimensional array")
            for i in range(array.shape[0]):
                try:
                    slice_df = pd.DataFrame(array[i])
                    slice_df.to_csv(f"{output_base}_slice{i}.csv", index=False)
                except ValueError:
                    # If conversion fails, save the raw data
                    np.savetxt(f"{output_base}_slice{i}.csv", array[i], delimiter=',')
            print(f"Saved {array.shape[0]} slices")
        
        else:
            # 1D array
            df = pd.DataFrame(array.reshape(-1, 1))
            df.to_csv(f"{output_base}.csv", index=False)
            print(f"Saved 1D array to {output_base}.csv")

def process_dict(data_dict, output_base):
    """Process a dictionary of arrays"""
    print(f"Processing dictionary with keys: {list(data_dict.keys())}")
    
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            process_array(value, f"{output_base}_{key}")
        elif isinstance(value, dict):
            process_dict(value, f"{output_base}_{key}")
        else:
            print(f"Skipping {key} with type {type(value)}")

def process_hdf5_dataset(dataset, output_base):
    """Process an HDF5 dataset (for MATLAB v7.3 files)"""
    if isinstance(dataset, h5py.Dataset):
        # Convert HDF5 dataset to numpy array
        try:
            array = np.array(dataset)
            process_array(array, output_base)
        except Exception as e:
            print(f"Error converting HDF5 dataset: {e}")
    elif isinstance(dataset, h5py.Group):
        # Process HDF5 group
        for key in dataset.keys():
            process_hdf5_dataset(dataset[key], f"{output_base}_{key}")

def save_complex_array(array, output_base):
    """Save a complex array that can't be directly converted to DataFrame"""
    try:
        # Try to flatten the array if possible
        flattened = np.hstack([col.flatten() for col in array.T])
        np.savetxt(f"{output_base}_flattened.csv", flattened, delimiter=',')
        print(f"Saved flattened array to {output_base}_flattened.csv")
    except:
        # If flattening fails, save raw data
        np.save(f"{output_base}.npy", array)
        print(f"Saved as NumPy binary file: {output_base}.npy")
        print("To load this file in Python: data = np.load(filename)")

# Example usage
if __name__ == "__main__":
    mat_file = r"E:\Machine Learning\CodE\1.Completed Projects\Depi Graduation project\data\5. Battery Data Set\B0005.mat"  # Replace with your actual file path
    convert_structured_mat(mat_file)