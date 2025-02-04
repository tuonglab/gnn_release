import pandas as pd
import csv
import os

def filter_scores(csv_file, id_filter_file, scores_file, output_file=None):
    """
    Filters the scores.txt file based on IDs from id_filter_file and data.csv.
    Outputs a comma-separated TXT file with the same prefix as the scores file.

    Args:
        csv_file (str): Path to the data.csv file.
        id_filter_file (str): Path to the filter_ids.txt file.
        scores_file (str): Path to the scores.txt file.
        output_file (str, optional): Path to save the filtered scores. 
                                   If None, generates a name based on scores_file. Defaults to None.
    """

    # 1. Read filter IDs
    with open(id_filter_file, 'r') as f:
        filter_ids = set(line.strip() for line in f)

    # 2. Read data.csv
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print(f"Error: {csv_file} is empty.")
        return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    valid_ids = set(df['id']) - filter_ids
    valid_sequences = set(df[df['id'].isin(valid_ids)]['sequence'])

    # 3. Determine output file name if not provided
    if output_file is None:
        scores_file_prefix = os.path.splitext(scores_file)[0]
        output_file = f"{scores_file_prefix}_filtered.txt"

    # 4. Filter scores.txt and write to output file
    try:
        with open(scores_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            # Use csv.Sniffer to detect the delimiter
            dialect = csv.Sniffer().sniff(infile.read(1024))
            infile.seek(0)  # Reset file pointer to the beginning
            reader = csv.reader(infile, dialect)
            writer = csv.writer(outfile, delimiter=',')  # Use comma as delimiter for output

            for line_number, row in enumerate(reader, 1):
                if not row or not "".join(row).strip():  # Skip empty lines or lines with only whitespace
                    continue

                if len(row) >= 2:
                    sequence = row[0]
                    score = row[1]

                    if sequence in valid_sequences:
                        writer.writerow([sequence, score])
                else:
                    print(f"Warning: Line {line_number} in {scores_file} has an unexpected format: '{','.join(row)}'")

    except Exception as e:
        print(f"Error reading or processing {scores_file}: {e}")
        return

    print(f"Filtering complete. Filtered scores saved to {output_file}")


# Example Usage:
csv_file = '/scratch/project/tcr_ml/colabfold/seekgene/N43_cdr3.csv'
id_filter_file = '/scratch/project/tcr_ml/gnn_release/seekgene_trial/N43_nontrb_sequences.txt'
scores_file = '/scratch/project/tcr_ml/gnn_release/seekgene_trial/seekgene_scores/N43_1_control_cdr3_scores.txt'
filter_scores(csv_file, id_filter_file, scores_file)