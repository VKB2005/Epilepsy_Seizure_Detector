# import os
# import glob
# import numpy as np

# PROCESSED_DIR = 'processed_data'
# OUTPUT_DIR = '.'

# if __name__ == "__main__":
#     print("Combining data (memory-efficiently)...")
    
#     x_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*_X.npy')))
#     y_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*_y.npy')))

#     # Calculate final shape
#     first_x = np.load(x_files[0])
#     num_samples, num_channels = first_x.shape[1], first_x.shape[2]
#     total_windows = sum(len(np.load(f)) for f in y_files)
#     final_x_shape = (total_windows, num_samples, num_channels)
#     print(f"Final X shape will be: {final_x_shape}")

#     # Create memory-mapped array on disk
#     output_x_path = os.path.join(OUTPUT_DIR, 'X_data.npy')
#     X_memmap = np.memmap(output_x_path, dtype=np.float32, mode='w+', shape=final_x_shape)

#     # Copy data chunk by chunk
#     print("Copying data chunks...")
#     current_pos = 0
#     for f in x_files:
#         chunk = np.load(f).astype(np.float32)
#         chunk_len = len(chunk)
#         X_memmap[current_pos : current_pos + chunk_len] = chunk
#         current_pos += chunk_len
    
#     X_memmap.flush()

#     # Combine Y files
#     print("Combining label files...")
#     all_y = [np.load(f) for f in y_files]
#     y = np.hstack(all_y)
#     output_y_path = os.path.join(OUTPUT_DIR, 'y_data.npy')
#     np.save(output_y_path, y)

#     print("\n--- Data Combination Complete! ---")