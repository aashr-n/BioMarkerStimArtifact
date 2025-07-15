import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def select_files():
    """Opens a dialog to select two CSV files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filepaths = filedialog.askopenfilenames(
        title="Select two CSV files (one 'stim_info', one 'arm1')",
        filetypes=[("CSV files", "*.csv")]
    )
    if len(filepaths) != 2:
        print("Error: Please select exactly two files.")
        return None
    
    if os.path.basename(filepaths[0])[:5] != os.path.basename(filepaths[1])[:5]:
        print("INCOMPATIBLE CSV FILES, MAKE SURE BOTH ARE FROM SAME PATIENT")
        raise SystemExit
    return filepaths

def identify_files(filepaths):
    """Identifies which file is 'stim_info' and which is 'arm1'."""
    stim_info_path = None
    arm1_path = None
    for path in filepaths:
        if "stim_info" in os.path.basename(path).lower():
            stim_info_path = path
        else:
            # Assume the other file is the arm1 file
            arm1_path = path
    
    if not stim_info_path or not arm1_path:
        print("Error: Could not identify both a 'stim_info' file and the other data file.")
        print("Please ensure one file has 'stim_info' in its name.")
        return None, None
        
    return stim_info_path, arm1_path

def merge_and_sort_data(stim_info_path, arm1_path):
    """
    Reads, merges, and sorts data from the two provided CSV files.
    """
    try:
        # Load the CSV files into pandas DataFrames
        stim_df = pd.read_csv(stim_info_path)
        arm1_df = pd.read_csv(arm1_path)
        print(f"Loaded {os.path.basename(stim_info_path)} with {len(stim_df)} rows.")
        print(f"Loaded {os.path.basename(arm1_path)} with {len(arm1_df)} rows.")

        # --- Data Preparation ---
        # Create the common timestamp column and the new 'type' column for each DataFrame.
        stim_df['sort_timestamp'] = pd.to_datetime(stim_df['stop_adj'])
        stim_df['type'] = 'stim'

        arm1_df['sort_timestamp'] = pd.to_datetime(arm1_df['stim_onoff_timestamp'])
        arm1_df['type'] = 'survey'

        # --- Merging ---
        # Concatenate the two DataFrames, keeping all original columns.
        # Mismatched columns will be filled with NaN.
        combined_df = pd.concat([stim_df, arm1_df], ignore_index=True, sort=False)

        # --- Sorting ---
        # Sort the combined DataFrame by the new timestamp column
        sorted_df = combined_df.sort_values(by='sort_timestamp', ascending=True)

        # --- Column Reordering ---
        # Get a list of all columns
        all_cols = sorted_df.columns.tolist()
        # Remove the columns we want to place at the front
        all_cols.remove('sort_timestamp')
        all_cols.remove('type')
        # Create the new column order and reorder the DataFrame
        new_order = ['sort_timestamp', 'type'] + all_cols
        final_df = sorted_df[new_order]
        
        return final_df

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except KeyError as e:
        print(f"Error: A required column is missing from a CSV file: {e}")
        print("Please check that 'stop_adj' is in the stim_info file and 'stim_onoff_timestamp' is in the arm1 file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def analyze_and_plot_stim_survey_pairs(merged_df, stim_info_path):
    """
    Finds consecutive stim-survey pairs, calculates the time difference,
    and creates several plots to visualize the data.
    """
    if merged_df is None or merged_df.empty:
        print("Dataframe is empty, cannot analyze pairs.")
        return

    # Get patient ID for plot titles from the first 5 chars of the stim file name
    try:
        patient_id = os.path.basename(stim_info_path)[:5]
    except:
        patient_id = "P"

    # Using pandas shift() to find rows where 'type' is 'stim' and the next row's 'type' is 'survey'.
    # .shift(-1) looks at the value in the next row.
    is_stim_followed_by_survey = (merged_df['type'] == 'stim') & (merged_df['type'].shift(-1) == 'survey')

    # Get the timestamps for the stim events that start a pair
    stim_times = merged_df.loc[is_stim_followed_by_survey, 'sort_timestamp']

    # Get the timestamps for the survey events that end a pair.
    # .shift(1) looks at the previous row. We find where the PREVIOUS row was the start of a pair.
    survey_times = merged_df.loc[is_stim_followed_by_survey.shift(1, fill_value=False), 'sort_timestamp']

    if stim_times.empty:
        print("\nNo consecutive stim-survey pairs were found in the data.")
        return

    # Reset indices to align the two series for calculation
    stim_times.reset_index(drop=True, inplace=True)
    survey_times.reset_index(drop=True, inplace=True)

    # Calculate the time differences
    time_deltas = survey_times - stim_times

    # Convert to seconds for plotting
    time_deltas_seconds = time_deltas.dt.total_seconds()

    print(f"\nFound {len(time_deltas_seconds)} stim-survey pairs.")

    # --- Plotting ---
    # Figure 1: Stacked Dot Plot (Strip Plot)
    plt.figure(figsize=(12, 7))
 
    # Round time differences to the nearest second for stacking
    rounded_times = np.round(time_deltas_seconds)
    time_counts = pd.Series(rounded_times).value_counts()

    # Create the plot data by stacking dots for each second
    plot_x = []
    plot_y = []
    for time_val, count in time_counts.items():
        plot_x.extend([time_val] * count)
        plot_y.extend(range(1, count + 1))

    plt.scatter(plot_x, plot_y, alpha=0.7)
 
    plt.title(f'({patient_id}) Instances of Time Delays from Stim to Survey')
    plt.xlabel('Time Difference (seconds)')
    plt.ylabel('Instance Count per Second')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Figure 2: Box Plot
    if not time_deltas_seconds.empty:
        plt.figure(figsize=(8, 6))
        plt.boxplot(time_deltas_seconds, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
        plt.title(f'({patient_id}) Distribution of Time Differences (Stim to Survey)')
        plt.ylabel('Time Difference (seconds)')
        plt.xticks([]) # Hide x-axis ticks as there's only one category
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()

    # Figure 3: Cumulative Frequency Line Plot
    if not time_deltas_seconds.empty:
        plt.figure(figsize=(12, 7))
        max_time = time_deltas_seconds.max()
        # Define thresholds in steps of 5 seconds, starting from 0
        thresholds = np.arange(0, max_time + 5, 5)
        # Calculate the number of instances below each threshold
        counts_below_threshold = [np.sum(time_deltas_seconds <= t) for t in thresholds]

        # The maximum number of instances is the total number of pairs found
        max_instances = len(time_deltas_seconds)

        # Create the line plot
        plt.plot(thresholds, counts_below_threshold, marker='o', linestyle='-', color='teal', label='Cumulative Count')

        # Add a horizontal red line to show the max number of instances
        plt.axhline(y=max_instances, color='r', linestyle='--', label=f'Total Instances: {max_instances}')

        plt.title(f'({patient_id}) Cumulative Count of Surveys Below Time Threshold')
        plt.xlabel('Time Difference Threshold (seconds)')
        plt.ylabel('Number of Instances Below Threshold')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Dynamically set x-axis ticks to prevent overcrowding
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20, integer=True))
        plt.xticks(rotation=45)

        plt.legend()
        plt.tight_layout()

    print("\nDisplaying plots. Close all plot windows to save the file and exit.")
    plt.show()

def save_file(dataframe, stim_info_path):
    """Opens a save dialog and saves the DataFrame to a CSV file."""
    if dataframe is None:
        print("No data to save due to an earlier error.")
        return
        
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Merged CSV File",
        initialfile=(f"{os.path.basename(stim_info_path)[:5]}_MERGED_timeline.csv")
    )
    
    if save_path:
        dataframe.to_csv(save_path, index=False)
        print(f"\nSuccessfully saved merged data to: {save_path}")
    else:
        print("\nSave operation cancelled.")

def main():
    """Main function to run the script."""
    filepaths = select_files()
    if not filepaths:
        return

    stim_info_path, arm1_path = identify_files(filepaths)
    if stim_info_path and arm1_path:
        merged_data = merge_and_sort_data(stim_info_path, arm1_path)
        # Analyze and plot the data before saving
        if merged_data is not None:
            # Pass stim_info_path to get the patient ID for titles
            analyze_and_plot_stim_survey_pairs(merged_data, stim_info_path)
        save_file(merged_data, stim_info_path)

if __name__ == "__main__":
    main()