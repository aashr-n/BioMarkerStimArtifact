import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# The time window before a stim must be clear of other stims for it to be "isolated".
BUFFER_SECONDS = 10
DURATION_SECONDS = 30

# The increment for testing different time thresholds on the plot.
THRESHOLD_STEP_SECONDS = 5
# ---------------------

def create_visualization(filepath):
    """
    Analyzes a CSV file to visualize the count of eligible stim-survey pairs
    against various time thresholds.
    """
    if not filepath:
        print("No file selected. Exiting.")
        return

    print(f"Processing file: {filepath}")
    
    # Read the data and ensure timestamp columns are converted to datetime objects
    try:
        df = pd.read_csv(
            filepath,
            parse_dates=['sort_timestamp', 'start_adj']
        )
    except (ValueError, KeyError) as e:
        messagebox.showerror(
            "File Error",
            f"Could not process the CSV file. Make sure it contains 'type', 'sort_timestamp', and 'start_adj' columns.\n\nError: {e}"
        )
        return

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
        'survey_start': df.loc[stim_indices + 1, 'sort_timestamp'].values
    })
    
    # Calculate the time delta for all potential pairs
    pairs_df['delta_seconds'] = (pairs_df['survey_start'] - pairs_df['stim_end']).dt.total_seconds()

    # 3. Identify which of the stimulation events are "isolated"
    # An isolated stim does not have another stim within the defined window before it.
    
    # Get all stim start times from the original dataframe, sorted chronologically
    all_stim_starts = df.loc[df['type'] == 'stim', 'start_adj'].sort_values().reset_index(drop=True)
    
    # Calculate the time difference between each stim and the one preceding it
    time_since_last_stim = all_stim_starts.diff()
    
    # Create a temporary dataframe to merge this info back into our pairs_df
    stim_isolation_df = pd.DataFrame({
        'stim_start': all_stim_starts,
        'time_since_last': time_since_last_stim
    })
    
    # Merge the isolation info. Use a 'left' merge to keep all pairs.
    pairs_df = pd.merge(pairs_df, stim_isolation_df, on='stim_start', how='left')
    
    # A stim is isolated if the time since the last one is > the required window,
    # or if it's the very first stim (time_since_last is NaT).
    isolation_window_seconds = BUFFER_SECONDS + DURATION_SECONDS
    is_isolated_mask = (pairs_df['time_since_last'].dt.total_seconds() > isolation_window_seconds) | (pairs_df['time_since_last'].isna())
    
    isolated_pairs_df = pairs_df[is_isolated_mask].copy()
    print(f"Identified {len(isolated_pairs_df)} isolated pairs out of {len(pairs_df)} total.")

    # 4. Calculate the counts for each threshold
    # Determine the maximum threshold to check based on the data
    max_delta = pairs_df['delta_seconds'].max()
    if pd.isna(max_delta):
        max_delta = 300 # Default max if no valid deltas found
        
    max_threshold = int(max_delta + THRESHOLD_STEP_SECONDS)
    thresholds = range(0, max_threshold, THRESHOLD_STEP_SECONDS)
    
    all_counts = []
    isolated_counts = []
    
    for threshold in thresholds:
        # Count all pairs within the current threshold
        all_count = (pairs_df['delta_seconds'] <= threshold).sum()
        all_counts.append(all_count)
        
        # Count isolated pairs within the current threshold
        isolated_count = (isolated_pairs_df['delta_seconds'] <= threshold).sum()
        isolated_counts.append(isolated_count)

    # 5. Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot for all eligible instances
    ax.plot(thresholds, all_counts, color='red', marker='o', linestyle='-', label='All Eligible Pairs')

    # Plot for isolated instances only
    ax.plot(thresholds, isolated_counts, color='blue', marker='x', linestyle='--', label=f'Isolated Pairs (No stim within {isolation_window_seconds}s before)')

    # Formatting the plot
    ax.set_title('Eligible Stim-Survey Pairs vs. Time Threshold', fontsize=16)
    ax.set_xlabel('Time Threshold (seconds)', fontsize=12)
    ax.set_ylabel('Number of Eligible Instances', fontsize=12)
    ax.legend(fontsize=10)
    ax.margins(x=0.02, y=0.05)
    
    # Ensure x-axis ticks are reasonable integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    plt.show()

    print("\nVisualization complete.")


def main():
    """Main function to trigger the file dialog and processing."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    filepath = filedialog.askopenfilename(
        title="Select the merged CSV file for visualization",
        filetypes=[("CSV files", "*.csv")]
    )
    
    create_visualization(filepath)


if __name__ == "__main__":
    main()
