"""
Split European_Bank.csv into two smaller CSV files:
- European_Bank_part1.csv
- European_Bank_part2.csv

Instructions for VS Code / Local PC:
1. Save this script (split_bank_csv.py) in the same folder as European_Bank.csv.
2. Open VS Code Terminal (Ctrl + ~) or your Command Prompt in that directory.
3. Run:
       python split_bank_csv.py
"""

import os
import pandas as pd

def split_csv(input_file="European_Bank.csv", output_part1="European_Bank_part1.csv", output_part2="European_Bank_part2.csv"):
    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find '{input_file}' in the current working directory.")
        print("Please place this script in the same folder as European_Bank.csv and try again.")
        return

    print(f"📖 Reading '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        total_rows = len(df)
        halfway = total_rows // 2

        part1 = df.iloc[:halfway]
        part2 = df.iloc[halfway:]

        part1.to_csv(output_part1, index=False)
        part2.to_csv(output_part2, index=False)

        print("\n✅ Splitting complete!")
        print(f"   • Total records in original file: {total_rows:,}")
        print(f"   • Created '{output_part1}' with {len(part1):,} rows")
        print(f"   • Created '{output_part2}' with {len(part2):,} rows")

    except Exception as e:
        print(f"⚠️ Pandas reading encountered an issue: {e}")
        print("🔄 Falling back to line-by-line raw splitting...")
        split_csv_raw(input_file, output_part1, output_part2)

def split_csv_raw(input_file, output_part1, output_part2):
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ Error: File is empty.")
        return

    header = lines[0]
    data_lines = lines[1:]

    halfway = len(data_lines) // 2
    part1_lines = [header] + data_lines[:halfway]
    part2_lines = [header] + data_lines[halfway:]

    with open(output_part1, "w", encoding="utf-8") as f1:
        f1.writelines(part1_lines)

    with open(output_part2, "w", encoding="utf-8") as f2:
        f2.writelines(part2_lines)

    print(f"✅ Raw split complete! Generated '{output_part1}' and '{output_part2}'.")

if __name__ == "__main__":
    split_csv()
