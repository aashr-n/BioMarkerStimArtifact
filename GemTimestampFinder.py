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
    and saves them to three CSV files. A fourth CSV is created to indicate
    whether each pair met the pre-stim quiet period requirement.
    """
    if not filepath:
        print("No file selected. Exiting.")
        return

    print(f"Processing file: {filepath}")
    input_path = Path(filepath)

    # Read the data and ensure required columns are present and correctly typed
    try:
        required_cols = ['record_id', 'sort_timestamp', 'start_adj', 'type']
        df = pd.read_csv(
            filepath,
            usecols=required_cols, # Only load columns we need
            parse_dates=['sort_timestamp', 'start_adj']
        )
    except (ValueError, KeyError) as e:
        messagebox.showerror(
            "File Error",
            f"Could not process the CSV file. Make sure it contains 'record_id', 'sort_timestamp', 'start_adj', and 'type' columns.\n\nError: {e}"
        )
        return

    # --- Calculate time since the previous stim for all stim events ---
    stims_only_df = df[df['type'] == 'stim'].copy()
    stims_only_df['time_since_prev_stim'] = (
        stims_only_df['start_adj'] - stims_only_df['sort_timestamp'].shift(1)
    ).dt.total_seconds()
    df['time_since_prev_stim'] = stims_only_df['time_since_prev_stim']

    # 1. Find all rows where a 'stim' type is directly followed by a 'survey' type
    is_stim_followed_by_survey = (df['type'] == 'stim') & (df['type'].shift(-1) == 'survey')
    stim_indices = df.index[is_stim_followed_by_survey]

    if stim_indices.empty:
        messagebox.showinfo("No Pairs Found", "No consecutive stim-survey pairs were found in the data.")
        return

    print(f"Found {len(stim_indices)} potential stim-survey pairs.")

    # 2. Create a DataFrame of the potential pairs
    pairs_df = pd.DataFrame({
        'stim_start': df.loc[stim_indices, 'start_adj'].values,
        'stim_end': df.loc[stim_indices, 'sort_timestamp'].values,
        'survey_start': df.loc[stim_indices + 1, 'sort_timestamp'].values,
        'time_since_prev_stim': df.loc[stim_indices, 'time_since_prev_stim'].values,
        'record_id': df.loc[stim_indices + 1, 'record_id'].values # Get record_id from the survey row
    })

    # 3. Filter pairs based on the survey time threshold
    pairs_df['delta_seconds'] = (pairs_df['survey_start'] - pairs_df['stim_end']).dt.total_seconds()
    valid_pairs_df = pairs_df[pairs_df['delta_seconds'] <= TIME_THRESHOLD_SECONDS].copy()

    if valid_pairs_df.empty:
        messagebox.showinfo(
            "No Valid Pairs",
            f"Found {len(pairs_df)} pairs, but none were within the {TIME_THRESHOLD_SECONDS} second survey threshold."
        )
        return

    print(f"Found {len(valid_pairs_df)} pairs within the survey time threshold.")

    # --- Determine if the quiet period was met for each valid pair ---
    quiet_period_seconds = 2 * (DURATION_SECONDS + BUFFER_SECONDS)
    # The first stim in the file will have NaN. fillna() ensures it's correctly marked as True.
    valid_pairs_df['quiet_period_met'] = (
        valid_pairs_df['time_since_prev_stim'].fillna(quiet_period_seconds + 1) > quiet_period_seconds
    )
    # --------------------------------------------------------------------

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

    # 5. Final formatting for the 'before', 'during', and 'after' DataFrames
    all_dfs = {
        'before': before_df,
        'during': during_df,
        'after': after_df
    }

    for name, temp_df in all_dfs.items():
        # Create the specific label format (e.g., "some_id_before")
        temp_df['label'] = valid_pairs_df['record_id'].astype(int).astype(str) + f"_{name}"

        # Reorder columns to put 'label' first.
        # The 'quiet_period_met' column is NOT included here.
        cols = ['label', 'start_time', 'end_time', 'adjust']
        all_dfs[name] = temp_df[cols]

    # --- NEW: Create a 4th DataFrame for the quiet time status ---
    quiet_status_df = pd.DataFrame({
        'label': valid_pairs_df['record_id'].astype(int).astype(str),
        'quiet_period_met': valid_pairs_df['quiet_period_met']
    })
    # ---------------------------------------------------------------

    # 6. Save the results to new CSV files
    output_dir = input_path.parent
    base_name = input_path.stem

    output_paths = {
        "before": output_dir / f"{base_name}_before.csv",
        "during": output_dir / f"{base_name}_during.csv",
        "after": output_dir / f"{base_name}_after.csv",
        "quiet_status": output_dir / f"{base_name}_quiet_status.csv" # Added 4th file path
    }

    datetime_format = '%Y-%m-%d %H:%M:%S.%f'

    # Save the original three files
    all_dfs['before'].to_csv(output_paths["before"], index=False, date_format=datetime_format)
    all_dfs['during'].to_csv(output_paths["during"], index=False, date_format=datetime_format)
    all_dfs['after'].to_csv(output_paths["after"], index=False, date_format=datetime_format)

    # Save the new quiet status file (no date format needed)
    quiet_status_df.to_csv(output_paths["quiet_status"], index=False)


    print("\n--- Success! ---")
    print(f"Output files have been saved to: {output_dir}")
    for name, path in output_paths.items():
        print(f"- {name.replace('_', ' ').capitalize()}: {path.name}")


def main():
    """Main function to trigger the file dialog and processing."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    filepath = filedialog.askopenfilename(
        title="Select the merged CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    if filepath: # Ensure a file was actually selected before processing
        process_stim_data(filepath)


if __name__ == "__main__":
    main()
