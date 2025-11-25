import os
import glob
import warnings
import mne
import numpy as np

# --- Configuration ---
DATA_DIR = '.'
SEIZURE_INFO_FILE = 'seizure_files.txt'
# New: A directory to store the intermediate processed files
PROCESSED_DIR = 'processed_data' 
# Create the directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- (Keep the parse_seizure_info and create_labeled_windows functions exactly the same as before) ---
def parse_seizure_info(file_path):
    seizure_info = {}
    with open(file_path, 'r') as f:
        current_filename = None
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                current_filename = os.path.basename(line.split(": ")[1])
                seizure_info[current_filename] = []
            elif line.startswith("Seizure Start Time:"):
                start_time = int(line.split(": ")[1].split(" ")[0])
            elif line.startswith("Seizure End Time:"):
                end_time = int(line.split(": ")[1].split(" ")[0])
                seizure_info[current_filename].append((start_time, end_time))
    return seizure_info

def create_labeled_windows(data, sfreq, seizure_info, window_seconds, overlap_ratio):
    window_samples = int(sfreq * window_seconds)
    step_samples = int(window_samples * (1 - overlap_ratio))
    windows, labels = [], []
    seizure_samples = [(int(start * sfreq), int(end * sfreq)) for start, end in seizure_info]
    start_idx = 0
    while start_idx + window_samples <= len(data):
        end_idx = start_idx + window_samples
        is_seizure = False
        for sei_start, sei_end in seizure_samples:
            if start_idx < sei_end and end_idx > sei_start:
                is_seizure = True
                break
        windows.append(data[start_idx:end_idx])
        labels.append(1 if is_seizure else 0)
        start_idx += step_samples
    return np.array(windows), np.array(labels)


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings('ignore', message="Channel names are not unique.*")
    
    print("STAGE 1: Processing files individually...")
    master_seizure_info = parse_seizure_info(SEIZURE_INFO_FILE)
    edf_files = glob.glob(os.path.join(DATA_DIR, '**', '*.edf'), recursive=True)
    
    processed_count = 0
    for i, file_path in enumerate(edf_files):
        filename = os.path.basename(file_path)
        print(f"  - Processing file {i+1}/{len(edf_files)}: {filename}")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # **FIX FOR ValueError**: Check the number of channels. Skip if not 23.
            if raw.info['nchan'] != 23:
                print(f"    -> Skipping {filename}: Found {raw.info['nchan']} channels, expected 23.")
                continue

            sfreq = raw.info['sfreq']
            eeg_data = raw.get_data().T
            seizure_times = master_seizure_info.get(filename, [])
            
            X_windows, y_labels = create_labeled_windows(
                data=eeg_data, sfreq=sfreq, seizure_info=seizure_times,
                window_seconds=1, overlap_ratio=0.5
            )
            
            # **FIX FOR MemoryError**: Save results for this file immediately.
            if X_windows.shape[0] > 0:
                output_basename = os.path.splitext(filename)[0]
                np.save(os.path.join(PROCESSED_DIR, f'{output_basename}_X.npy'), X_windows)
                np.save(os.path.join(PROCESSED_DIR, f'{output_basename}_y.npy'), y_labels)
                processed_count += 1

        except Exception as e:
            print(f"    -> Error processing {filename}: {e}")

    print(f"\n--- Stage 1 Complete! ---")
    print(f"Successfully processed and saved {processed_count} files to the '{PROCESSED_DIR}' folder.")