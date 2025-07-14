import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

# --- Configuration ---
BUFFER_SECONDS = 10
DURATION_SECONDS = 30
THRESHOLD_STEP_SECONDS = 5
SURVEY_TIME_THRESHOLD_SECONDS = 90

def get_plot_title(filepath):
    """Extracts 'RCSXX' from a filename to create a dynamic plot title."""
    match = re.search(r'RCS\d{2}', str(filepath), re.IGNORECASE)
    return match.group(0) if match else "Analysis"

def load_and_validate_data(filepath):
    """Loads CSV, validates required columns, and handles potential duplicates."""
    try:
        df = pd.read_csv(filepath, parse_dates=['sort_timestamp', 'start_adj'])
        df.rename(columns={'sort_timestamp': 'stop_adj'}, inplace=True)
    except Exception as e:
        messagebox.showerror("File Read Error", f"Could not read or parse the CSV file.\nError: {e}")
        return None

    required_cols = ['type', 'start_adj', 'stop_adj']
    for col in required_cols:
        if col not in df.columns:
            messagebox.showerror("Missing Column", f"The required column '{col}' was not found in the file.")
            return None
    
    if isinstance(df['start_adj'], pd.DataFrame) or isinstance(df['stop_adj'], pd.DataFrame):
        print("Warning: Duplicate column names found. The script will attempt to use the first instance of each.")
        
    return df

def find_stim_survey_pairs(df):
    """Identifies stim-survey pairs and returns a clean DataFrame."""
    is_stim_followed_by_survey = (df['type'] == 'stim') & (df['type'].shift(-1) == 'survey')
    stim_indices = df.index[is_stim_followed_by_survey]
    
    if stim_indices.empty:
        return None

    stim_starts = df.loc[stim_indices, 'start_adj']
    stim_ends = df.loc[stim_indices, 'stop_adj']
    survey_starts = df.loc[stim_indices + 1, 'stop_adj']

    if isinstance(stim_starts, pd.DataFrame): stim_starts = stim_starts.iloc[:, 0]
    if isinstance(stim_ends, pd.DataFrame): stim_ends = stim_ends.iloc[:, 0]
    if isinstance(survey_starts, pd.DataFrame): survey_starts = survey_starts.iloc[:, 0]

    pairs_df = pd.DataFrame({
        'stim_start': stim_starts.reset_index(drop=True),
        'stim_end': stim_ends.reset_index(drop=True),
        'survey_start': survey_starts.reset_index(drop=True)
    })
    
    pairs_df['delta_seconds'] = (pairs_df['survey_start'] - pairs_df['stim_end']).dt.total_seconds()
    return pairs_df

def plot_figure_1(pairs_df, all_stim_starts, plot_id):
    """Generates and displays the 'Eligible vs. Isolated Pairs' plot."""
    print("Generating Figure 1: Eligible vs. Isolated Pairs...")
    time_since_last_stim = all_stim_starts.diff().dt.total_seconds()
    stim_isolation_df = pd.DataFrame({'stim_start': all_stim_starts, 'time_since_last': time_since_last_stim})
    
    pairs_with_isolation = pd.merge(pairs_df, stim_isolation_df, on='stim_start', how='left')
    
    isolation_window_seconds = BUFFER_SECONDS + DURATION_SECONDS
    is_isolated_mask = (pairs_with_isolation['time_since_last'] > isolation_window_seconds) | (pairs_with_isolation['time_since_last'].isna())
    isolated_pairs_df = pairs_with_isolation[is_isolated_mask]

    max_delta = pairs_with_isolation['delta_seconds'].max()
    max_threshold = int(max_delta + THRESHOLD_STEP_SECONDS) if pd.notna(max_delta) else 300
    thresholds = range(0, max_threshold, THRESHOLD_STEP_SECONDS)
    
    all_counts = [(pairs_with_isolation['delta_seconds'] <= t).sum() for t in thresholds]
    isolated_counts = [(isolated_pairs_df['delta_seconds'] <= t).sum() for t in thresholds]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(thresholds, all_counts, color='red', marker='o', linestyle='-', label='All Eligible Pairs')
    ax.plot(thresholds, isolated_counts, color='blue', marker='o', linestyle='--', label=f'Isolated Pairs (No stim within {isolation_window_seconds}s before)')
    ax.set_title(f'{plot_id}: Eligible Stim-Survey Pairs vs. Time Threshold', fontsize=16)
    ax.set_xlabel('Time Threshold (seconds)', fontsize=12)
    ax.set_ylabel('Number of Eligible Instances', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    return fig

def plot_figure_2(pairs_df, stim_periods, plot_id):
    """Generates and displays the 'Proximity to No-Stim Windows' plot."""
    print("Generating Figure 2: Proximity to No-Stim Windows...")
    no_stim_window_seconds = DURATION_SECONDS + BUFFER_SECONDS

    # --- FIX ---
    # The previous error occurred here because duplicate columns in `stim_periods`
    # caused `gap_starts` and `gap_ends` to be 2D DataFrames instead of 1D Series.
    # This fix ensures they are always 1D Series before creating the new DataFrame.
    gap_starts = stim_periods['stop_adj']
    gap_ends = stim_periods['start_adj'].shift(-1)

    if isinstance(gap_starts, pd.DataFrame): gap_starts = gap_starts.iloc[:, 0]
    if isinstance(gap_ends, pd.DataFrame): gap_ends = gap_ends.iloc[:, 0]
    
    gaps = pd.DataFrame({
        'start': gap_starts.reset_index(drop=True), 
        'end': gap_ends.reset_index(drop=True)
    }).dropna()
    # --- END FIX ---

    gaps['duration'] = (gaps['end'] - gaps['start']).dt.total_seconds()
    
    valid_gaps = gaps[gaps['duration'] >= no_stim_window_seconds].reset_index(drop=True)

    if valid_gaps.empty:
        print("No valid no-stim windows found. Skipping Figure 2.")
        return None

    pairs_sorted = pairs_df.sort_values('stim_start')
    merged_prev = pd.merge_asof(pairs_sorted, valid_gaps.rename(columns={'end': 'prev_gap_end'}),
                                left_on='stim_start', right_on='prev_gap_end', direction='backward')
    dist_to_prev = (merged_prev['stim_start'] - merged_prev['prev_gap_end']).dt.total_seconds()

    merged_next = pd.merge_asof(pairs_sorted, valid_gaps.rename(columns={'start': 'next_gap_start'}),
                                left_on='stim_start', right_on='next_gap_start', direction='forward')
    dist_to_next = (merged_next['next_gap_start'] - merged_next['stim_start']).dt.total_seconds()
    
    pairs_df['dist_to_prev'] = dist_to_prev
    pairs_df['dist_to_next'] = dist_to_next
    pairs_df['nearest_dist'] = pairs_df[['dist_to_prev', 'dist_to_next']].min(axis=1)
    pairs_df['no_stim_precedes'] = pairs_df['dist_to_prev'] <= pairs_df['dist_to_next']

    pairs_df_90s = pairs_df[pairs_df['delta_seconds'] <= SURVEY_TIME_THRESHOLD_SECONDS].copy()
    max_dist = pairs_df['nearest_dist'].max()
    dist_thresholds = range(0, int(max_dist + THRESHOLD_STEP_SECONDS) if pd.notna(max_dist) else 300, THRESHOLD_STEP_SECONDS)

    counts_A = [(pairs_df['nearest_dist'] <= t).sum() for t in dist_thresholds]
    counts_B = [((pairs_df['nearest_dist'] <= t) & (~pairs_df['no_stim_precedes'])).sum() for t in dist_thresholds]
    counts_C = [(pairs_df_90s['nearest_dist'] <= t).sum() for t in dist_thresholds]
    counts_D = [((pairs_df_90s['nearest_dist'] <= t) & (~pairs_df_90s['no_stim_precedes'])).sum() for t in dist_thresholds]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(dist_thresholds, counts_A, color='green', marker='o', label='All Pairs')
    ax.plot(dist_thresholds, counts_B, color='orange', marker='o', linestyle='--', label='Pairs w/o Preceding No-Stim')
    ax.plot(dist_thresholds, counts_C, color='purple', marker='s', label=f'Pairs within {SURVEY_TIME_THRESHOLD_SECONDS}s Threshold')
    ax.plot(dist_thresholds, counts_D, color='brown', marker='s', linestyle='--', label=f'Pairs w/o Preceding No-Stim (within {SURVEY_TIME_THRESHOLD_SECONDS}s)')

    ax.set_title(f'{plot_id}: Instance Proximity to Eligible No-Stim Windows', fontsize=16)
    ax.set_xlabel('Distance to Nearest No-Stim Window (seconds)', fontsize=12)
    ax.set_ylabel('Number of Instances', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    return fig

def main():
    """Main function to orchestrate the data processing and visualization."""
    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename(
        title="Select the merged CSV file for visualization",
        filetypes=[("CSV files", "*.csv")]
    )
    if not filepath:
        print("No file selected. Exiting.")
        return

    plot_id = get_plot_title(Path(filepath).name)
    df = load_and_validate_data(filepath)
    if df is None:
        return

    pairs_df = find_stim_survey_pairs(df)
    if pairs_df is None:
        messagebox.showinfo("No Pairs Found", "No consecutive stim-survey pairs were found in the data.")
        return

    # --- Generate Plots ---
    all_stim_starts_raw = df.loc[df['type'] == 'stim', 'start_adj']
    if isinstance(all_stim_starts_raw, pd.DataFrame): all_stim_starts_raw = all_stim_starts_raw.iloc[:, 0]
    all_stim_starts = all_stim_starts_raw.sort_values().reset_index(drop=True)

    stim_periods_raw = df[df['type'] == 'stim'][['start_adj', 'stop_adj']]
    # This is more complex, we need to handle duplicates for each column individually
    start_adj_raw = stim_periods_raw['start_adj']
    stop_adj_raw = stim_periods_raw['stop_adj']
    if isinstance(start_adj_raw, pd.DataFrame): start_adj_raw = start_adj_raw.iloc[:, 0]
    if isinstance(stop_adj_raw, pd.DataFrame): stop_adj_raw = stop_adj_raw.iloc[:, 0]
    stim_periods = pd.DataFrame({'start_adj': start_adj_raw, 'stop_adj': stop_adj_raw}).sort_values('start_adj').reset_index(drop=True)


    fig1 = plot_figure_1(pairs_df, all_stim_starts, plot_id)
    fig2 = plot_figure_2(pairs_df, stim_periods, plot_id)

    plt.tight_layout()
    plt.show()
    print("\nVisualization complete.")

if __name__ == "__main__":
    main()
