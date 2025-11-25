import os

# The base URL for the dataset on PhysioNet
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

# The input file containing the list of seizure records
INPUT_FILE = "seizure_files.txt"

# The output script that will contain all the download commands
OUTPUT_SCRIPT_NAME = "download_seizure_data.sh"

print(f"Reading file list from: {INPUT_FILE}")

files_to_download = set()

# Open the file with 'rU' mode for universal newline support
with open(INPUT_FILE, 'r') as f:
    for line in f:
        # Strip all leading/trailing whitespace including \r and \n
        file_path = line.strip()
        
        # Make sure the line is not empty and is an .edf file
        if file_path and file_path.endswith(".edf"):
            files_to_download.add(file_path)

print(f"Found {len(files_to_download)} unique seizure files to download.")

# Create the download script
with open(OUTPUT_SCRIPT_NAME, 'w', newline='\n') as f: # Use Unix-style line endings
    f.write("#!/bin/bash\n")
    f.write("echo 'Starting download...'\n\n")

    for file_path in sorted(list(files_to_download)):
        directory = os.path.dirname(file_path)
        full_url = BASE_URL + file_path
        
        # Command to create directory if it doesn't exist, then download
        command = f"mkdir -p {directory}\nwget -c -N -P {directory} {full_url}\n"
        f.write(command)

    f.write("\necho 'Download complete!'\n")

print(f"Success! A download script named '{OUTPUT_SCRIPT_NAME}' has been created.")
print("Now, you can run this script from your terminal.")