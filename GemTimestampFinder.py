import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd

# --- Configuration ---
# Threshold in seconds for a survey to be considered "following" a stim.
# A delta less than or equal to this value is included.
TIME_THRESHOLD_SECONDS = 300

# Buffer period in seconds to place between stim and the before/after windows.
BUFFER_SECONDS = 10

# Duration in seconds for each of the before, during, and after windows.
DURATION_SECONDS = 30
# ---------------------


def process_stim_data(filepath):
    """
    Identifies stim-survey pairs, calculates time windows (before, during, after),
    and saves them to separate CSV files.
    """
    if not filepath:
        print("No file selected. Exiting.")
        return

    print(f"Processing file: {filepath}")
    input_path = Path(filepath)
    
    # Read the data and ensure timestamp columns are converted to datetime objects
    try:
        df = pd.read_csv(
            filepath,
            parse_dates=['sort_timestamp', 'start_adj']
        )
    except (ValueError, KeyError) as e:
        messagebox.showerror(
            "File Error",
            f"Could not process the CSV file. Make sure it contains 'sort_timestamp' and 'start_adj' columns.\n\nError: {e}"
        )
        return

    # 1. Find all rows where a 'stim' type is directly followed by a 'survey' type
    is_stim_followed_by_survey = (df['type'] == 'stim') & (df['type'].shift(-1) == 'survey')
    stim_indices = df.index[is_stim_followed_by_survey]
    
    if stim_indices.empty:
        messagebox.showinfo("No Pairs Found", "No consecutive stim-survey pairs were found in the data.")
        return
        
    print(f"Found {len(stim_indices)} potential stim-survey pairs.")

    # 2. Create a DataFrame of the valid pairs
    pairs_df = pd.DataFrame({
        'stim_start': df.loc[stim_indices, 'start_adj'].values,
        'stim_end': df.loc[stim_indices, 'sort_timestamp'].values,
        'survey_start': df.loc[stim_indices + 1, 'sort_timestamp'].values
    })

    # 3. Filter pairs based on the time threshold
    pairs_df['delta_seconds'] = (pairs_df['survey_start'] - pairs_df['stim_end']).dt.total_seconds()
    valid_pairs_df = pairs_df[pairs_df['delta_seconds'] <= TIME_THRESHOLD_SECONDS].copy()

    if valid_pairs_df.empty:
        messagebox.showinfo(
            "No Valid Pairs",
            f"Found {len(pairs_df)} pairs, but none were within the {TIME_THRESHOLD_SECONDS} second threshold."
        )
        return

    print(f"Found {len(valid_pairs_df)} pairs within the {TIME_THRESHOLD_SECONDS} second threshold.")

    # 4. Calculate the 'before', 'during', and 'after' time windows
    buffer_delta = pd.to_timedelta(BUFFER_SECONDS, unit='s')
    duration_delta = pd.to_timedelta(DURATION_SECONDS, unit='s')
    
    # --- Before Stim ---
    before_df = pd.DataFrame()
    before_df['start_time'] = valid_pairs_df['stim_start'] - buffer_delta - duration_delta
    before_df['end_time'] = valid_pairs_df['stim_start'] - buffer_delta
    before_df['adjust'] = 'auto'

    # --- During Stim ---
    stim_total_duration = valid_pairs_df['stim_end'] - valid_pairs_df['stim_start']
    stim_midpoint = valid_pairs_df['stim_start'] + (stim_total_duration / 2)
    during_df = pd.DataFrame()
    during_df['start_time'] = stim_midpoint - (duration_delta / 2)
    during_df['end_time'] = stim_midpoint + (duration_delta / 2)
    during_df['adjust'] = 'auto'

    # --- After Stim ---
    after_df = pd.DataFrame()
    after_df['start_time'] = valid_pairs_df['stim_end'] + buffer_delta
    after_df['end_time'] = valid_pairs_df['stim_end'] + buffer_delta + duration_delta
    after_df['adjust'] = 'auto'

    # 5. Final formatting for all three DataFrames
    all_dfs = [before_df, during_df, after_df]
    for temp_df in all_dfs:
        # Round times to the nearest second
        temp_df['start_time'] = temp_df['start_time'].dt.round('S')
        temp_df['end_time'] = temp_df['end_time'].dt.round('S')
        
        # Add a 1-based index as the 'label' column at the front
        temp_df.insert(0, 'label', temp_df.index + 1)


    # 6. Save the results to new CSV files
    output_dir = input_path.parent
    base_name = input_path.stem

    output_paths = {
        "before": output_dir / f"{base_name}_before.csv",
        "during": output_dir / f"{base_name}_during.csv",
        "after": output_dir / f"{base_name}_after.csv",
    }

    datetime_format = '%Y-%m-%d %H:%M:%S'
    
    before_df.to_csv(output_paths["before"], index=False, date_format=datetime_format)
    during_df.to_csv(output_paths["during"], index=False, date_format=datetime_format)
    after_df.to_csv(output_paths["after"], index=False, date_format=datetime_format)
    
    print("\n--- Success! ---")
    print(f"Output files have been saved to: {output_dir}")
    for name, path in output_paths.items():
        print(f"- {name.capitalize()}: {path.name}")
        
    messagebox.showinfo(
        "Success",
        f"Processing complete. The output files have been saved to:\n\n{output_dir}"
    )


def main():
    """Main function to trigger the file dialog and processing."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    filepath = filedialog.askopenfilename(
        title="Select the merged CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    
    process_stim_data(filepath)


if __name__ == "__main__":
    main()