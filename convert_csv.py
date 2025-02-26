import os
import csv
from pathlib import Path

def convert_to_csv(input_dir, output_dir):
    """
    Convert CDR3 text files to CSV format with sequence IDs.
    
    Args:
        input_dir (str): Directory containing the *_cdr3.txt files
        output_dir (str): Directory where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each cdr3.txt file
    for file in os.scandir(input_dir):
        if file.name.endswith('.txt'):
            input_file = Path(file.path)
            output_file = Path(output_dir) / f"{input_file.stem}.csv"
            
            print(f"Processing {file.name}...")
            
            try:
                # Read sequences from text file
                with open(input_file, 'r') as f:
                    sequences = [line.strip() for line in f if line.strip()]
                
                # Write to CSV with IDs
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(['id', 'sequence'])
                    
                    # Write sequences with IDs
                    for idx, seq in enumerate(sequences, 1):
                        writer.writerow([f"seq_{idx}", seq])
                
                print(f"Successfully created {output_file}")
                print(f"Processed {len(sequences)} sequences")
                
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")

if __name__ == "__main__":
    # Replace these paths with your actual paths
    input_directory = "/QRISdata/Q7250/ZERO/sarcoma/tcr_cdr3"
    output_directory = "/scratch/project/tcr_ml/colabfold/sarcoma_zero"
    
    convert_to_csv(input_directory, output_directory)