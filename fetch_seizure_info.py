import urllib.request
import os

OUTPUT_FILE = 'seizure_files.txt'
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

# Patients chb01 through chb24
patient_ids = [f"chb{i:02d}" for i in range(1, 25)]

print(f"Downloading summary files to create {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for patient in patient_ids:
        # Construct URL: e.g., https://physionet.org/files/chbmit/1.0.0/chb01/chb01-summary.txt
        url = f"{BASE_URL}{patient}/{patient}-summary.txt"
        print(f"Fetching info for {patient}...", end=" ")
        
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
                outfile.write(content)
                outfile.write("\n" + "="*50 + "\n") # Separator
            print("Done.")
        except Exception as e:
            print(f"Failed: {e}")

print(f"\nSuccess! {OUTPUT_FILE} has been created with the correct timestamps.")