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
    Identifies stim-survey pairs with a valid pre-stim quiet period,
    calculates time windows (before, during, after), and saves them
    to separate CSV files.
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
    # --- FIX 1: Added 'time_since_prev_stim' to this DataFrame ---
    pairs_df = pd.DataFrame({
        'stim_start': df.loc[stim_indices, 'start_adj'].values,
        'stim_end': df.loc[stim_indices, 'sort_timestamp'].values,
        'survey_start': df.loc[stim_indices + 1, 'sort_timestamp'].values,
        'time_since_prev_stim': df.loc[stim_indices, 'time_since_prev_stim'].values
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

    # --- Filter pairs based on the pre-stim "quiet period" ---
    quiet_period_seconds = 2*(DURATION_SECONDS + BUFFER_SECONDS)
    final_pairs_df = valid_pairs_df[
        valid_pairs_df['time_since_prev_stim'].fillna(quiet_period_seconds + 1) > quiet_period_seconds
    ].copy()

    if final_pairs_df.empty:
        messagebox.showinfo(
            "No Valid Pairs",
            f"Found {len(valid_pairs_df)} pairs within the survey threshold, but none had a sufficient "
            f"pre-stimulation quiet period of {quiet_period_seconds} seconds."
        )
        return

    print(f"Found {len(final_pairs_df)} pairs with a sufficient pre-stim quiet period.")

    # 4. Calculate the 'before', 'during', and 'after' time windows
    # --- FIX 2: Changed all calculations to use 'final_pairs_df' ---
    buffer_delta = pd.to_timedelta(BUFFER_SECONDS, unit='s')
    duration_delta = pd.to_timedelta(DURATION_SECONDS, unit='s')

    # --- Before Stim ---
    before_df = pd.DataFrame()
    before_df['start_time'] = final_pairs_df['stim_start'] - buffer_delta - duration_delta
    before_df['end_time'] = final_pairs_df['stim_start'] - buffer_delta
    before_df['adjust'] = 'auto'

    # --- During Stim ---
    stim_total_duration = final_pairs_df['stim_end'] - final_pairs_df['stim_start']
    stim_midpoint = final_pairs_df['stim_start'] + (stim_total_duration / 2)
    during_df = pd.DataFrame()
    during_df['start_time'] = stim_midpoint - (duration_delta / 2)
    during_df['end_time'] = stim_midpoint + (duration_delta / 2)
    during_df['adjust'] = 'auto'

    # --- After Stim ---
    after_df = pd.DataFrame()
    after_df['start_time'] = final_pairs_df['stim_end'] + buffer_delta
    after_df['end_time'] = final_pairs_df['stim_end'] + buffer_delta + duration_delta
    after_df['adjust'] = 'auto'

    # 5. Final formatting for all three DataFrames
    all_dfs = [before_df, during_df, after_df]
    for temp_df in all_dfs:
        # --- FIX 3: Re-added reset_index for sequential labels ---
        temp_df.reset_index(drop=True, inplace=True)
        temp_df.insert(0, 'label', temp_df.index + 1)


    # 6. Save the results to new CSV files
    output_dir = input_path.parent
    base_name = input_path.stem

    output_paths = {
        "before": output_dir / f"{base_name}_before.csv",
        "during": output_dir / f"{base_name}_during.csv",
        "after": output_dir / f"{base_name}_after.csv",
    }

    datetime_format = '%Y-%m-%d %H:%M:%S.%f'

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
