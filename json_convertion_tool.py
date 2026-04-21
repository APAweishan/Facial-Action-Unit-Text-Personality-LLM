import pandas as pd
import json
import argparse

def convert_excel_to_json(excel_file, json_file, id_column='id'):
    keep_columns = [
        id_column,
        'gender', 'age', 'education', 'work_experience',
        'Honesty-Humility', 'Extraversion', 'Agreeableness', 'Conscientiousness'
    ]

    # Read Excel files
    df = pd.read_csv(excel_file)

    # Check if all required columns exist
    missing_cols = [col for col in keep_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following are missing from Excel: {missing_cols}")

    # Filter the columns to keep
    df = df[keep_columns]

    # Set the id column as the index
    df = df.set_index(id_column)

    # Convert to dictionary
    data_dict = df.to_dict(orient='index')

    # Write to JSON file
    with open(json_file, 'w') as f:
        json.dump(data_dict, f, indent=2)

    print(f"Successfully converted {excel_file} to {json_file}")
    print(f"Retained fields include: {keep_columns[1:]}")
    print(f"Total entries: {len(data_dict)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', type=str, required=True, help='Path: Input Excel file')
    parser.add_argument('--json_file', type=str, required=True, help='Path: Output JSON file')
    parser.add_argument('--id_column', type=str, default='id', help='Column name for unique sample identifier, default is "id"')
    args = parser.parse_args()

    convert_excel_to_json(args.excel_file, args.json_file, args.id_column)
